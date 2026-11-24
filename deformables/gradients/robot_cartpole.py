"""Matched CPU pilot on force-Jacobian supervision for the public SuperNeRD cart-pole MLP."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmark import (
    DT,
    FORCE_N,
    Dynamics,
    NewtonCartpole,
    controls,
    random_state,
    teacher_trace,
    wrapped_error,
)
from calibrate_cartpole import CalibratedCartpole

DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[2]
EPS = 0.004
STEPS = 750
BATCH = 256
ARMS = ("values", "extra_values", "force_jacobian")
SEEDS = (3, 17, 41)
JAC_WEIGHT = 0.05
now = time.perf_counter


def center_bank(teacher, seed, horizon, kind):
    n = teacher.n
    rng = np.random.default_rng(seed)
    if kind == "train":
        state = random_state(rng, n, [3.85, math.pi, 6, 8], DEVICE)
        state[n // 2 :] = random_state(rng, n - n // 2, [0.8, 0.5, 1, 1.2], DEVICE)
    elif kind == "passive":
        state = random_state(rng, n, [1, math.pi, 1, 1], DEVICE)
    else:
        state = random_state(rng, n, [0.8, 0.4, 0.7, 1], DEVICE)
    action = (
        controls(rng, n, horizon, DEVICE)
        if kind != "passive"
        else torch.zeros(horizon, n)
    )
    if kind == "train":
        action[:4, : n // 4] = torch.from_numpy(
            rng.uniform(-0.15, 0.15, (4, n // 4)).astype(np.float32)
        )
    return teacher_trace(teacher, state, action), action


def perturb(teacher, state, action, h):
    plus, minus = [], []
    for s, a in zip(state[:-1], action, strict=True):
        teacher.reset(s)
        plus.append(teacher(a + h))
        teacher.reset(s)
        minus.append(teacher(a - h))
    plus, minus = torch.stack(plus), torch.stack(minus)
    return plus, minus, (plus.double() - minus.double()).float() / (2 * h)


def neural_jacobian(model, state, action, train):
    a = action.detach().requires_grad_()
    u = model(state, a)
    return torch.stack(
        [
            torch.autograd.grad(
                u[:, i].sum(), a, create_graph=train, retain_graph=True
            )[0]
            for i in range(4)
        ],
        1,
    )


def train(arm, seed, bank):
    torch.manual_seed(seed)
    model = Dynamics()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=0.0002)
    rng = torch.Generator().manual_seed(seed + 900)
    s, a, p, m, j = bank
    jac_scale = j.square().mean((0, 1)).sqrt().clamp_min(1e-6)
    scale = model.state_drift_scale
    start = now()
    for step in range(STEPS):
        h = 1 if step < 0.8 * STEPS else 4
        t = torch.randint(len(a) - h + 1, (BATCH,), generator=rng)
        rows = torch.randint(a.shape[1], (BATCH,), generator=rng)
        x, action = s[t, rows], a[t, rows]
        pred = x
        loss = 0
        for k in range(h):
            pred = model(pred, a[t + k, rows])
            loss += (
                (pred - s[t + k + 1, rows]) / ((k + 1) * scale)
            ).square().mean() / h
        if arm == "extra_values":
            plus = ((model(x, action + EPS) - p[t, rows]) / scale).square().mean()
            minus = ((model(x, action - EPS) - m[t, rows]) / scale).square().mean()
            loss = (loss + (plus + minus) / 2) / 2
        if arm == "force_jacobian":
            jac = neural_jacobian(model, x, action, True)
            loss += JAC_WEIGHT * ((jac - j[t, rows]) / jac_scale).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
    return model.eval(), now() - start


def score(model, banks, derivative):
    dtype = next(model.parameters()).dtype
    truth, action, jac = derivative
    predicted = (
        neural_jacobian(model, truth.to(dtype), action.to(dtype), False)
        .detach()
        .double()
    )
    err = (predicted - jac).square().sum(0)
    norms = jac.square().sum(0)
    scores = {
        "heldout_one_step_action_jacobian_relative_rms": (err / norms).sqrt().tolist()
    }
    for case, (states, actions) in banks.items():
        with torch.no_grad():
            s = states[0].to(dtype)
            errors = []
            for t, a in enumerate(actions):
                s = model(s, a.to(dtype))
                errors.append(wrapped_error(s, states[t + 1].to(dtype)))
            e = torch.stack(errors)
            scores[case] = {
                str(h): {
                    "whole_window_cart_rmse_m": e[:h, :, 0]
                    .square()
                    .mean()
                    .sqrt()
                    .item(),
                    "whole_window_angle_rmse_deg": torch.rad2deg(e[:h, :, 1])
                    .square()
                    .mean()
                    .sqrt()
                    .item(),
                    "last_angle_rmse_deg": torch.rad2deg(e[h - 1, :, 1])
                    .square()
                    .mean()
                    .sqrt()
                    .item(),
                }
                for h in (1, 15, 60, 100)
            }
    return scores


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--urdf", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tic = now()
    teacher = NewtonCartpole(args.urdf, 192, DEVICE)
    setup = now() - tic
    tic = now()
    state, action = center_bank(teacher, 0, 32, "train")
    capture = now() - tic
    tic = now()
    p, m, j = perturb(teacher, state, action, EPS)
    fd_time = now() - tic
    bank = (state, action, p, m, j)
    test = {
        case: center_bank(teacher, 12016 + i, 100, case)
        for i, case in enumerate(("active", "passive"))
    }
    sj, aj = center_bank(teacher, 734049, 6, "train")
    _, _, j1 = perturb(teacher, sj, aj, EPS / 2)
    _, _, j2 = perturb(teacher, sj, aj, EPS / 4)
    score_bank = (sj[:-1].flatten(0, 1), aj.flatten(), j2.flatten(0, 1).double())
    report = {
        "dt_s": DT,
        "newtons_per_action_unit": FORCE_N,
        "train_central_difference_halfwidth": EPS,
        "derivative_channels": [
            "cart position",
            "angle",
            "cart velocity",
            "angular velocity",
        ],
        "training_center_cases": 192,
        "central_teacher_steps": action.numel(),
        "additional_teacher_steps": 2 * action.numel(),
        "data_seed": 0,
        "test_forced_trajectory_seed": 12016,
        "test_passive_trajectory_seed": 12017,
        "test_derivative_seed": 734049,
        "test_derivative_cases": j1.numel() // 4,
        "heldout_halfwidth": EPS / 4,
        "heldout_reference_relative_discrepancy_halfwidth_2x": (
            (j1 - j2).square().sum((0, 1)) / j2.square().sum((0, 1))
        )
        .sqrt()
        .tolist(),
        "setup_cpu_s": setup,
        "base_capture_s": capture,
        "extra_capture_s": fd_time,
        "optimizer_updates": STEPS,
        "batch_size": BATCH,
        "jacobian_relative_loss_weight": JAC_WEIGHT,
        "geometry_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "fits": {},
    }
    calibrated = CalibratedCartpole().eval()
    calibrated.load_state_dict(
        torch.load(
            ROOT / "weights/calibrated-modern-common.pt",
            weights_only=True,
            map_location="cpu",
        )
    )
    report["published_calibrated_six_parameter_reference"] = score(
        calibrated, test, score_bank
    )
    published = Dynamics().eval()
    published.load_state_dict(
        torch.load(ROOT / "weights/model.pt", weights_only=True, map_location="cpu")
    )
    reference = score(published, test, score_bank)
    (args.output / "reference-mlp.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    for seed in SEEDS:
        report["fits"][str(seed)] = {}
        for arm in ARMS:
            model, elapsed = train(arm, seed, bank)
            metrics = score(model, test, score_bank)
            report["fits"][str(seed)][arm] = {"fit_cpu_s": elapsed, **metrics}
            print(seed, arm, round(elapsed, 2), metrics["active"]["100"], flush=True)
            torch.save(model.state_dict(), args.output / f"{arm}-{seed}.pt")
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    fields = {}
    for arm in ARMS:
        rows = [report["fits"][str(seed)][arm] for seed in SEEDS]
        fields[arm] = {
            "mean_fit_s": np.mean([r["fit_cpu_s"] for r in rows]),
            "mean_test_force_jacobian_relative_rms": np.mean(
                [r["heldout_one_step_action_jacobian_relative_rms"] for r in rows],
                axis=0,
            ).tolist(),
            "mean_active_100_angle_rmse_deg": np.mean(
                [r["active"]["100"]["whole_window_angle_rmse_deg"] for r in rows]
            ),
            "mean_passive_100_angle_rmse_deg": np.mean(
                [r["passive"]["100"]["whole_window_angle_rmse_deg"] for r in rows]
            ),
        }
    report["means_over_three_seeds"] = fields
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
