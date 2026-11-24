"""Matched continuation on broad Newton loads with or without small-perturbation rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from gradients.force_gradient_trial import loss_for_batch, sample_fields
from gradients.local_fit import KEYS, initial_model, load_fit
from gradients.local_fit import evaluate as score_local
from modal_forward import load, sync
from native_rollouts import evaluate as score_native

SEEDS = (271, 725)
STEPS = 1200


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sample_batches(seed, old, local, device):
    original_rng = torch.Generator(device=device).manual_seed(seed)
    local_rng = torch.Generator(device=device).manual_seed(seed + 11423)
    cases = torch.randint(
        len(old["q"]), (STEPS, 24), generator=original_rng, device=device
    )
    starts = torch.cat(
        [
            torch.randint(301 - h, (n, 24), generator=original_rng, device=device)
            for n, h in ((STEPS // 3, 140), (STEPS - STEPS // 3, 200))
        ]
    )
    starts[:, :6] = 0
    tiny_cases = torch.randint(
        len(local["q0"]), (STEPS, 12), generator=local_rng, device=device
    )
    digest = hashlib.sha256(
        torch.stack((cases, starts)).cpu().numpy().tobytes()
    ).hexdigest()
    return cases, starts, tiny_cases, digest


def local_secant_loss(model, inputs, ids, levels):
    q0, v0, static, controls, direction, target = (inputs[key][ids] for key in KEYS)
    probes = (controls[:, None] + levels * direction[:, None]).flatten(0, 1)
    predicted = model(
        *(x.repeat_interleave(3, 0) for x in (q0, v0, static)), probes
    ).view_as(target)
    shape_loss = ((predicted - target).square().mean((-1, -2)) + 1e-7).sqrt().mean()
    true_difference = target[:, 2] - target[:, 0]
    pred_difference = predicted[:, 2] - predicted[:, 0]
    error = (pred_difference - true_difference).square().mean((1, 2, 3))
    scale = true_difference.square().mean((1, 2, 3)).sqrt().clamp_min(0.0003)
    return shape_loss + 0.03 * ((error + 1e-12).sqrt() / scale).mean()


def fit(arm, seed, old, alternate, local, args, device):
    model = initial_model(args.parent, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00012, fused=device.type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, STEPS, eta_min=0.00002
    )
    cases, starts, local_cases, digest = sample_batches(seed, old, local, device)
    levels = torch.tensor([-0.5, 0.0, 0.5], device=device)[None, :, None, None]
    sync(device)
    begin = time.perf_counter()
    for step in range(STEPS):
        horizon = 140 if step < STEPS // 3 else 200
        batch = sample_fields(old, alternate, cases[step], starts[step], horizon)
        old_loss = loss_for_batch(model, batch, False)
        if arm == "mixed":
            loss = 0.75 * old_loss + 0.25 * local_secant_loss(
                model, local, local_cases[step], levels
            )
        else:
            loss = old_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
    sync(device)
    seconds = time.perf_counter() - begin
    name = f"{arm}-{seed}.pt"
    torch.save(model.state_dict(), args.output / name)
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": name,
        "fit_seconds": seconds,
        "updates": STEPS,
        "sampled_original_indices_sha256": digest,
    }


@torch.no_grad()
def score_bank(model, packed, native):
    result = {}
    for start in (0, 100):
        inputs = [
            packed["q"][:, start],
            packed["v"][:, start],
            packed["static"],
            packed["u"][:, start:],
        ]
        output = model(*(torch.from_numpy(x) for x in inputs)).numpy()
        result[str(start)] = score_native(packed, native, output, start)
    return result


def read_npz(path):
    with np.load(path) as record:
        return dict(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "score"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation-data", type=Path)
    parser.add_argument("--confirmation-native", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if args.command == "fit":
        device = torch.device("cuda")
        a, b = (load(args.data / n, device) for n in ("train.npz", "train-pulses.npz"))
        assert np.array_equal(a["seed"], b["seed"]) and np.array_equal(
            a["family"], b["family"]
        )
        assert torch.equal(a["static"], b["static"]) and torch.equal(
            a["q"][:, 0], b["q"][:, 0]
        )
        local = load_fit(args.local, device)
        # Exercise every kernel shape before measuring either arm. These weights are discarded.
        dummy = initial_model(args.parent, device)
        ids = torch.arange(24, device=device)
        for horizon in (140, 200):
            loss_for_batch(
                dummy, sample_fields(a, b, ids, torch.zeros_like(ids), horizon), False
            ).backward()
            dummy.zero_grad(set_to_none=True)
        local_secant_loss(
            dummy,
            local,
            ids[:12],
            torch.tensor([-0.5, 0, 0.5], device=device)[None, :, None, None],
        ).backward()
        del dummy
        runs = []
        for seed, order in ((SEEDS[0], ("old", "mixed")), (SEEDS[1], ("mixed", "old"))):
            for arm in order:
                row = fit(arm, seed, a, b, local, args, device)
                print(row, flush=True)
                runs.append(row)
            assert (
                runs[-1]["sampled_original_indices_sha256"]
                == runs[-2]["sampled_original_indices_sha256"]
            )
        save_json(
            {
                "fits": runs,
                "device": torch.cuda.get_device_name(),
                "parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                "old_weight_on_mixed": 0.75,
                "local_weight_on_mixed": 0.25,
                "initial_lr": 0.00012,
                "final_lr": 0.00002,
                "matmul_precision": "highest",
            },
            args.output / "frozen.json",
        )
        return
    with (args.output / "frozen.json").open() as file:
        runs = json.load(file)["fits"]
    assert len(runs) == 4
    probes = [read_npz(p) for p in sorted((args.local / "probe").glob("*.npz"))]
    assert len(probes) == 16
    banks = {
        "fresh": (
            read_npz(args.data / "fresh.npz"),
            read_npz(args.data / "native-fresh.npz"),
        )
    }
    if args.confirmation_data and args.confirmation_native:
        banks["confirmation"] = (
            read_npz(args.confirmation_data),
            read_npz(args.confirmation_native),
        )
        train = read_npz(args.data / "train.npz")
        probe_seeds = {int(p["seed"]) for p in probes}
        confirm_seeds = set(banks["confirmation"][0]["seed"].tolist())
        assert confirm_seeds.isdisjoint(
            set(train["seed"])
        ) and confirm_seeds.isdisjoint(probe_seeds)
        assert confirm_seeds.isdisjoint(set(banks["fresh"][0]["seed"]))
    records = []
    for row in [{"arm": "parent", "seed": None, "checkpoint": args.parent.name}] + runs:
        name = row["arm"] if row["arm"] == "parent" else f"{row['arm']}-{row['seed']}"
        net = initial_model(
            args.parent if name == "parent" else args.output / row["checkpoint"], "cpu"
        ).eval()
        local_result = score_local(net, probes)
        save_json(local_result, args.output / f"{name}-local.json")
        result = {**row, "local": local_result["summary"]}
        for bank, (data, native) in banks.items():
            full = score_bank(net, data, native)
            result[bank] = {k: v["summary"] for k, v in full.items()}
            save_json(full, args.output / f"{name}-{bank}.json")
        records.append(result)
    save_json(records, args.output / "scores.json")


if __name__ == "__main__":
    main()
