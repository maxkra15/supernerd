"""Compare a paired-rollout loss with midpoint force-Jacobian supervision.

Run this script alongside modal_forward.py and native_rollouts.py. The midpoint
target is a finite Newton displacement along a line between two force programs.
It estimates a local derivative only to the degree that the response is linear
over that interval. Each JVP is taken from the same observed release state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from modal_forward import ModalForward, load, mean_dev, sync
from native_rollouts import evaluate as evaluate_native


def to_file(value, path):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def midpoint_jvp(model, q, v, static, program_a, program_b):
    midpoint = (program_a + program_b) * 0.5
    direction = program_a - program_b

    def predict(u):
        return model(q, v, static, u)

    return torch.func.jvp(predict, (midpoint,), (direction,))[1]


def sample_fields(primary, alternate, cases, starts, horizon):
    indices = starts[:, None] + torch.arange(horizon, device=cases.device)
    q, v, u, targets = [], [], [], []
    for dataset in (primary, alternate):
        q.append(dataset["q"][cases, starts])
        v.append(dataset["v"][cases, starts])
        u.append(dataset["u"][cases[:, None], indices])
        targets.append(dataset["q"][cases[:, None], indices + 1])
    static = primary["static"][cases].repeat(2, 1)
    return torch.cat(q), torch.cat(v), static, torch.cat(u), torch.cat(targets)


def loss_for_batch(model, fields, use_jvp):
    q, v, static, u, targets = fields
    pairs = len(q) // 2
    release = pairs // 4
    prediction = model(q, v, static, u)
    error = prediction - targets
    abs_loss = (error.square().mean(dim=(-1, -2)) + 1e-7).sqrt().mean()
    velocity_loss = (
        (error[:, 1:] - error[:, :-1])
        .square()
        .mean(dim=(-1, -2))
        .add(1e-8)
        .sqrt()
        .mean()
    )
    motion = targets[:pairs] - targets[pairs:]
    motion_error = error[:pairs] - error[pairs:]
    if use_jvp:
        tangent = midpoint_jvp(
            model,
            q[:release],
            v[:release],
            static[:release],
            u[:release],
            u[pairs : pairs + release],
        )
        motion_error = torch.cat((tangent - motion[:release], motion_error[release:]))
    scale = motion.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(0.001)
    response_loss = (
        (motion_error.square().mean(dim=(1, 2, 3)) + 1e-10).sqrt() / scale
    ).mean()
    return abs_loss + 0.7 * velocity_loss + 0.06 * response_loss


def check_derivative(primary, alternate):
    torch.manual_seed(91)
    model = ModalForward().cpu().double()
    q = primary["q"][:2, 0].cpu().double()
    v = primary["v"][:2, 0].cpu().double()
    static = primary["static"][:2].cpu().double()
    model.static_mean.copy_(primary["static"].mean(0).cpu())
    model.static_std.copy_(primary["static"].std(0).cpu().clamp_min(1e-6))
    a = primary["u"][:2, :130].cpu().double()
    b = alternate["u"][:2, :130].cpu().double()
    midpoint, direction = 0.5 * (a + b), a - b
    jvp = midpoint_jvp(model, q, v, static, a, b)
    step = 1e-3
    finite = (
        model(q, v, static, midpoint + step * direction)
        - model(q, v, static, midpoint - step * direction)
    ) / (2 * step)
    relative = (finite - jvp).norm() / jvp.norm()
    layer = model.readout[0].weight
    (weight_jvp,) = torch.autograd.grad(jvp.square().mean(), layer)
    (weight_fd,) = torch.autograd.grad(finite.square().mean(), layer)
    weight_relative = (weight_fd - weight_jvp).norm() / weight_jvp.norm()
    assert relative < 1e-5 and weight_relative < 1e-5
    return {
        "direction_relative_l2": float(relative.detach()),
        "weight_gradient_relative_l2": float(weight_relative.detach()),
        "centered_check_step": step,
        "dtype": "float64",
    }


def train(primary, alternate, args, arm, wall_budget):
    device = primary["q"].device
    torch.manual_seed(args.seed)
    model = ModalForward().to(device)
    model.static_mean.copy_(primary["static"].mean(dim=0))
    model.static_std.copy_(primary["static"].std(dim=0).clamp_min(1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, fused=device.type == "cuda")
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=2e-4)
    path = args.output / arm
    path.mkdir(parents=True, exist_ok=True)
    marks = {1, args.steps // 8, args.steps // 4, args.steps // 2, args.steps}
    records = []
    matched = None
    wall_saved = False
    # Reinitialize this seed for each arm; JVP and bookkeeping consume no random numbers.
    first_inputs, last_inputs = None, None
    sync(device)
    start = time.perf_counter()
    excluded = 0.0
    for step in range(1, args.steps + 1):
        horizon = 140 if step <= args.steps // 3 else 200
        cases = torch.randint(len(primary["q"]), (24,), device=device)
        starts = torch.randint(301 - horizon, (24,), device=device)
        starts[:6] = 0
        fields = sample_fields(primary, alternate, cases, starts, horizon)
        loss = loss_for_batch(model, fields, arm == "jvp")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        schedule.step()
        if step in marks or step % args.clock_interval == 0:
            sync(device)
            now = time.perf_counter()
            elapsed = now - start - excluded
            if step in (1, args.steps):
                digest = hashlib.sha256(
                    torch.stack((cases, starts)).cpu().numpy().tobytes()
                ).hexdigest()
                if step == 1:
                    first_inputs = digest
                if step == args.steps:
                    last_inputs = digest
            record = {"step": step, "fit_seconds": elapsed}
            if step in marks:
                name = f"step-{step}.pt"
                torch.save(model.state_dict(), path / name)
                record["checkpoint"] = name
                records.append(record.copy())
                print(arm, step, round(elapsed, 3), flush=True)
            if wall_budget is not None and not wall_saved:
                if elapsed <= wall_budget:
                    torch.save(model.state_dict(), path / "equal-time.pt")
                    matched = {**record, "checkpoint": "equal-time.pt"}
                else:
                    wall_saved = True
            sync(device)
            excluded += time.perf_counter() - now
    if matched:
        records.append(matched)
    return {
        "checkpoints": records,
        "final_fit_seconds": elapsed,
        "seed": args.seed,
        "first_cases_and_starts_sha256": first_inputs,
        "last_cases_and_starts_sha256": last_inputs,
    }


@torch.no_grad()
def score(args, reports, device):
    dev = load(args.data / "development.npz", device)
    for arm, report in reports.items():
        for row in report["checkpoints"]:
            model = ModalForward().to(device).eval()
            model.load_state_dict(
                torch.load(
                    args.output / arm / row["checkpoint"],
                    weights_only=True,
                    map_location=device,
                )
            )
            row["development"] = mean_dev(model, dev)
            if (
                row["step"] not in (args.steps,)
                and row["checkpoint"] != "equal-time.pt"
            ):
                continue
            for name in ("test", "fresh"):
                data = load(args.data / f"{name}.npz", device)
                with np.load(args.data / f"{name}.npz") as f:
                    packed = dict(f)
                with np.load(args.data / f"native-{name}.npz") as f:
                    native = dict(f)
                row[name] = {}
                for start in (0, 100):
                    pred = (
                        model(
                            data["q"][:, start],
                            data["v"][:, start],
                            data["static"],
                            data["u"][:, start:],
                        )
                        .cpu()
                        .numpy()
                    )
                    result = evaluate_native(packed, native, pred, start)
                    row[name][str(start)] = result["summary"]
                    to_file(
                        result,
                        args.output / arm / f"{row['checkpoint']}-{name}-{start}.json",
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--clock-interval", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    a, b = (load(args.data / n, device) for n in ("train.npz", "train-pulses.npz"))
    for k in ("family", "seed"):
        assert np.array_equal(a[k], b[k])
    for k in ("q", "v"):
        assert torch.equal(a[k][:, 0], b[k][:, 0])
    assert torch.equal(a["static"], b["static"])
    check = check_derivative(a, b)
    print("float64 checks", check, flush=True)
    # Warm all paths on a disposable model; both timed arms start from the common seed.
    tiny = ModalForward().to(device)
    ids = torch.arange(24, device=device)
    for h in (140, 200):
        for jvp in (False, True):
            tiny.zero_grad()
            loss_for_batch(
                tiny, sample_fields(a, b, ids, torch.zeros_like(ids), h), jvp
            ).backward()
    del tiny
    reports = {}
    for arm in ("paired", "jvp"):
        reports[arm] = train(
            a, b, args, arm, reports.get("paired", {}).get("final_fit_seconds")
        )
        to_file(reports, args.output / "progress.json")
    assert (
        reports["paired"]["first_cases_and_starts_sha256"]
        == reports["jvp"]["first_cases_and_starts_sha256"]
    )
    assert (
        reports["paired"]["last_cases_and_starts_sha256"]
        == reports["jvp"]["last_cases_and_starts_sha256"]
    )
    score(args, reports, device)
    to_file(
        {
            "arms": reports,
            "derivative_check": check,
            "hardware": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "checkpoint_time_granularity_updates": args.clock_interval,
            "timing": "Warm optimizer updates including synchronization, excluding setup, checkpoint I/O and all scoring.",
        },
        args.output / "results.json",
    )


if __name__ == "__main__":
    main()
