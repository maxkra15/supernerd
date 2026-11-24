"""Compare position, secant, and local derivative supervision on identical Newton recordings."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from modal_forward import ModalForward
from native_rollouts import decode, frame_error

ARMS = ("positions", "secant", "autodiff")
SEEDS = (271, 725, 1709)
STEPS = 1200
KEYS = ("q0", "v0", "static", "u", "direction", "q")


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def load_fit(root, device):
    names = sorted((root / "fit").glob("*.npz"))
    assert len(names) == 64
    rows = []
    for path in names:
        with np.load(path) as record:
            assert np.array_equal(record["amplitude"], [-0.5, 0, 0.5])
            rows.append({key: record[key] for key in KEYS})
    return {key: tensor(np.stack([row[key] for row in rows]), device) for key in KEYS}


def initial_model(weights, device):
    model = ModalForward().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model


def forward_tangent(model, q0, v0, static, u, direction):
    return torch.func.jvp(
        lambda loads: model(q0, v0, static, loads), (u,), (direction,)
    )[1]


def fit(args, train, arm, seed, device):
    torch.manual_seed(seed)
    model = initial_model(args.parent, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00012, fused=device.type == "cuda"
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, STEPS, eta_min=0.00002
    )
    levels = tensor([-0.5, 0, 0.5], device)[None, :, None, None]
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    sync()
    started = time.perf_counter()
    for step in range(STEPS):
        ids = torch.randint(len(train["q0"]), (12,), device=device)
        q0, v0, static, u, direction, target = [train[k][ids] for k in KEYS]
        inputs = (u[:, None] + levels * direction[:, None]).flatten(0, 1)
        pred = model(
            *(x.repeat_interleave(3, 0) for x in (q0, v0, static)), inputs
        ).view_as(target)
        error = pred - target
        value_loss = (error.square().mean(dim=(-1, -2)) + 1e-7).sqrt().mean()
        loss = value_loss
        if arm != "positions":
            target_slope = target[:, 2] - target[:, 0]  # denominator 2h is 1 m/s².
            predicted_slope = (
                pred[:, 2] - pred[:, 0]
                if arm == "secant"
                else forward_tangent(model, q0, v0, static, u, direction)
            )
            slope_error = (predicted_slope - target_slope).square().mean(dim=(1, 2, 3))
            scale = target_slope.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(0.0003)
            loss = loss + 0.03 * ((slope_error + 1e-12).sqrt() / scale).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        schedule.step()
    sync()
    seconds = time.perf_counter() - started
    checkpoint = args.output / f"{arm}-{seed}.pt"
    torch.save(model.state_dict(), checkpoint)
    return {
        "arm": arm,
        "seed": seed,
        "updates": STEPS,
        "fit_s": seconds,
        "checkpoint": checkpoint.name,
    }


def relative(pred, actual):
    return float(100 * np.linalg.norm(pred - actual) / np.linalg.norm(actual))


@torch.no_grad()
def evaluate(model, probes):
    rows = []
    for case in probes:
        levels = case["amplitude"].tolist()
        assert levels == [-0.5, -0.25, 0.0, 0.25, 0.5, 3.0]
        q0, v0, static, u, direction = [tensor(case[k][None], "cpu") for k in KEYS[:-1]]
        rest, free = case["rest"], case["free"].reshape(-1)

        def native(pred, static_value=case["static"], rest=rest, free=free):
            return decode(pred.detach().numpy()[0], static_value, rest, free)

        pred0 = native(model(q0, v0, static, u))
        pred3 = native(model(q0, v0, static, u + 3 * direction))
        tangent = native(forward_tangent(model, q0, v0, static, u, direction)) - rest
        actual = case["native"].astype(np.float64)
        onset = int(np.flatnonzero(np.linalg.norm(case["direction"], axis=1) > 1e-6)[0])
        kept = slice(onset, None)
        near = (actual[3] - actual[1])[kept] / 0.5
        far = (actual[4] - actual[0])[kept]
        estimate = tangent[kept].astype(np.float64)
        shape = frame_error(pred3 - actual[5])
        norm = np.linalg.norm(near)
        cosine = (near * estimate).sum() / max(norm * np.linalg.norm(estimate), 1e-30)
        rows.append(
            {
                "family": str(case["family"]),
                "seed": int(case["seed"]),
                "amplitude_3_shape_pct": float(100 * shape.mean() / case["static"][1]),
                "amplitude_3_response_pct": relative(
                    (pred3 - pred0)[kept], (actual[5] - actual[2])[kept]
                ),
                "local_derivative_error_pct": relative(estimate, near),
                "local_derivative_cos": float(cosine),
                "teacher_halving_disagreement_pct": relative(far, near),
                "teacher_halving_first_second_pct": relative(far[:50], near[:50]),
                "local_derivative_first_second_error_pct": relative(
                    estimate[:50], near[:50]
                ),
            }
        )
    metrics = [k for k in rows[0] if k not in ("family", "seed")]
    return {
        "cases": rows,
        "summary": {
            f: {
                k: float(np.mean([r[k] for r in rows if r["family"] == f]))
                for k in metrics
            }
            for f in ("cable", "cloth")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=Path("weights/modal-paired.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    times = args.output / "training.json"
    if not args.score_only:
        device = torch.device(args.device)
        train = load_fit(args.data, device)
        timings = []
        for i, seed in enumerate(SEEDS):
            for arm in ARMS[i:] + ARMS[:i]:
                run = fit(args, train, arm, seed, device)
                timings.append(run)
                print(run, flush=True)
                times.write_text(json.dumps(timings, indent=2) + "\n")
    timings = json.loads(times.read_text())
    test = []
    for path in sorted((args.data / "probe").glob("*.npz")):
        with np.load(path) as z:
            test.append(dict(z))
    assert len(test) == 16
    records = [
        {
            "arm": "parent",
            "checkpoint": args.parent.name,
            "seed": None,
            **evaluate(initial_model(args.parent, "cpu").eval(), test),
        }
    ]
    for run in timings:
        model = initial_model(args.output / run["checkpoint"], "cpu").eval()
        records.append(run | evaluate(model, test))
    (args.output / "results.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    for record in records:
        print(record["arm"], record["seed"], record["summary"], flush=True)


if __name__ == "__main__":
    main()
