"""Test whether paired force interventions improve forward response fidelity."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from modal_forward import ModalForward, evaluate, load, mean_dev, sync
from pack_deformables import pack


def repack(source, destination):
    records = sorted(
        json.loads((source / "manifest.json").read_text())["episodes"],
        key=lambda r: (r["family"], r["seed"], r["identity"]),
    )
    samples = []
    for record in records:
        path = source / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Teacher trajectory checksum mismatch")
        with np.load(path) as data:
            samples.append(pack(record, data))
    np.savez_compressed(
        destination,
        **{key: np.stack([sample[key] for sample in samples]) for key in samples[0]},
    )


def fit(data, alternate, dev, steps, device):
    torch.manual_seed(91)
    model = ModalForward().to(device)
    model.static_mean.copy_(data["static"].mean(dim=0))
    model.static_std.copy_(data["static"].std(dim=0).clamp_min(1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, fused=device.type == "cuda")
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=2e-4)
    curve = []
    excluded = 0.0
    sync(device)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        batch_size = 24
        horizon = 140 if step <= steps // 3 else 200
        cases = torch.randint(len(data["q"]), (batch_size,), device=device)
        starts = torch.randint(301 - horizon, (batch_size,), device=device)
        starts[: batch_size // 4] = 0
        time_index = starts[:, None] + torch.arange(horizon, device=device)
        fields = {}
        for key in ("q", "v", "static", "u"):
            if key in ("q", "v"):
                fields[key] = torch.cat(
                    [set_[key][cases, starts] for set_ in (data, alternate)]
                )
            elif key == "static":
                fields[key] = data[key][cases].repeat(2, 1)
            else:
                fields[key] = torch.cat(
                    [
                        set_[key][cases[:, None], time_index]
                        for set_ in (data, alternate)
                    ]
                )
        predicted = model(fields["q"], fields["v"], fields["static"], fields["u"])
        target = torch.cat(
            [set_["q"][cases[:, None], time_index + 1] for set_ in (data, alternate)]
        )
        error = predicted - target
        # Balance absolute position and small applied-force-induced trajectory changes.
        abs_loss = (error.square().mean(dim=(-1, -2)) + 1e-7).sqrt().mean()
        motion = target[:batch_size] - target[batch_size:]
        motion_error = error[:batch_size] - error[batch_size:]
        scale = motion.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(0.001)
        response_loss = (
            (motion_error.square().mean(dim=(1, 2, 3)) + 1e-10).sqrt() / scale
        ).mean()
        velocity = (
            (error[:, 1:] - error[:, :-1])
            .square()
            .mean(dim=(-1, -2))
            .add(1e-8)
            .sqrt()
            .mean()
        )
        loss = abs_loss + 0.7 * velocity + 0.06 * response_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        schedule.step()
        if step in {1, steps // 8, steps // 4, steps // 2, steps}:
            sync(device)
            checked = time.perf_counter()
            curve.append(
                {
                    "updates": step,
                    "fit_s": checked - start - excluded,
                    "dev": mean_dev(model, dev),
                    "paired_training_relative_loss": float(response_loss.detach()),
                }
            )
            sync(device)
            excluded += time.perf_counter() - checked
    return model.eval(), curve


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("data"))
    cli.add_argument("--paired-source", type=Path)
    cli.add_argument("--fresh-source", type=Path)
    cli.add_argument("--output", type=Path, default=Path("paired"))
    cli.add_argument("--steps", type=int, default=8000)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.paired_source:
        repack(args.paired_source, args.data / "train-pulses.npz")
    if args.fresh_source:
        repack(args.fresh_source, args.data / "fresh.npz")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    data = load(args.data / "train.npz", device)
    alternate = load(args.data / "train-pulses.npz", device)
    dev = load(args.data / "development.npz", device)
    assert np.array_equal(data["family"], alternate["family"]) and np.array_equal(
        data["seed"], alternate["seed"]
    )
    assert torch.equal(data["static"], alternate["static"]) and torch.equal(
        data["q"][:, 0], alternate["q"][:, 0]
    )
    model, curve = fit(data, alternate, dev, args.steps, device)
    report = {
        "model": "same ModalForward with matched-pair intervention loss",
        "seed": 91,
        "fit_episodes": len(data["q"]) * 2,
        "fit_objects": len(data["q"]),
        "updates": args.steps,
        "paired_loss_weight": 0.06,
        "fit_curve": curve,
        "eval": {},
    }
    for split in ("test", "fresh"):
        test = load(args.data / f"{split}.npz", device)
        assert set(data["seed"]).isdisjoint(test["seed"])
        report["eval"][split] = {}
        predictions = {"identity": test["identity"]}
        for t in (0, 100):
            report["eval"][split][str(t)], pred = evaluate(model, test, t, True)
            predictions[f"q_from_{t}"] = pred.cpu().numpy()
        np.savez_compressed(args.output / f"{split}-predictions.npz", **predictions)
    torch.save(model.state_dict(), args.output / "weights.pt")
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(curve[-1], flush=True)


if __name__ == "__main__":
    main()
