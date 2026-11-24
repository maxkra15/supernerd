"""Identify six positive Cartpole coefficients from independent Newton transitions."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from benchmark import DT, FORCE_N, SUBSTEPS, NewtonCartpole, controls, random_state, teacher_trace
from structured_dynamics import score
from torch import nn


class CalibratedCartpole(nn.Module):
    """Train pair inertia and rail forces while retaining a five-substep integrator."""

    def __init__(self):
        super().__init__()
        mass, inertia, coupling = 6.0, 1.1, 1.0
        rho = coupling / math.sqrt(mass * inertia)
        self.log_values = nn.Parameter(
            torch.tensor(
                [math.log(mass), math.log(inertia), math.log(7), math.log(6000), math.log(18)],
                dtype=torch.float64,
            )
        )
        self.coupling_logit = nn.Parameter(torch.tensor(math.log(rho / (1 - rho)), dtype=torch.float64))

    def coefficients(self):
        mass, inertia, gravity, stiffness, damping = self.log_values.exp().unbind()
        coupling = (mass * inertia).sqrt() * self.coupling_logit.sigmoid()
        return mass, inertia, coupling, gravity, stiffness, damping

    def forward(self, state, action):
        mass, inertia, coupling, gravity, stiffness, damping = self.coefficients()
        x, theta, vx, omega = state.unbind(dim=-1)
        applied = FORCE_N * action.reshape_as(x)
        off_diag = coupling * theta.cos()
        inv_det = (mass * inertia - off_diag.square()).reciprocal()
        dt = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            sin = theta.sin()
            outside = x - x.clamp(-4.0, 4.0)
            force_x = applied + coupling * sin * omega.square() - stiffness * outside - damping * vx * (outside != 0)
            torque = coupling * gravity * sin
            vx = vx + dt * (inertia * force_x - off_diag * torque) * inv_det
            omega = omega + dt * (mass * torque - off_diag * force_x) * inv_det
            x = x + dt * vx
            theta = theta + dt * omega
        return torch.stack((x, theta, vx, omega), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-inertias", action="store_true")
    parser.add_argument("--comparison-fit", type=Path, help="Fit JSON with a shared benchmark normalization")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    rng = np.random.default_rng(142)
    started = time.perf_counter()
    n = 1024
    teacher = NewtonCartpole(args.urdf, n, device, legacy_inertias=args.legacy_inertias)
    initial = random_state(rng, n, [3.92, math.pi, 6.0, 9.0], device)
    actions = controls(rng, n, 32, device)
    states = teacher_trace(teacher, initial, actions)
    stage_seconds = time.perf_counter() - started
    norm = states.std(dim=(0, 1)).clamp_min(1e-6)
    inputs = states[:-1].flatten(0, 1).double()
    target = states[1:].flatten(0, 1).double()
    actions = actions.flatten().double()
    # Preserve both the regular dynamics and the rail-limit rows in the optimizer sample.
    rail = torch.where(inputs[:, 0].abs() > 3.98)[0]
    base = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(9))[:4096]
    rows = torch.unique(torch.cat((base, rail)))
    inputs, target, actions = inputs[rows], target[rows], actions[rows]
    scale = torch.tensor([0.005, 0.008, 0.5, 0.8], dtype=torch.float64)
    model = CalibratedCartpole()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-13,
        tolerance_change=1e-15,
    )
    losses = []
    keys = (
        "effective_translation_mass_kg",
        "pole_inertia_kg_m2",
        "coupling_kg_m",
        "gravity_m_s2",
        "rail_stiffness_n_m",
        "rail_damping_n_s_m",
    )

    def values():
        return dict(zip(keys, (p.item() for p in model.coefficients()), strict=True))

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(inputs, actions) - target) / scale).square().mean()
        loss.backward()
        losses.append(loss.item())
        return loss

    before = values()
    fit_start = time.perf_counter()
    optimizer.step(closure)
    fit_s = time.perf_counter() - fit_start
    after = values()
    with torch.no_grad():
        residual = ((model(inputs, actions) - target) / scale).square().mean().item()
    with np.load(args.traces) as held_out:
        if bool(held_out.get("legacy_inertias", False)) != args.legacy_inertias:
            raise ValueError("Calibration and held-out asset inertia conventions differ")
    result = {
        "legacy_inertias": args.legacy_inertias,
        "initial_coefficients": before,
        "learned_coefficients": after,
        "fit_transitions_generated": len(states[:-1].flatten(0, 1)),
        "fit_transitions_selected": len(rows),
        "fit_rail_rows": len(rail),
        "fit_seed": 142,
        "fit_s": fit_s,
        "teacher_setup_and_data_s": stage_seconds,
        "closure_evaluations": len(losses),
        "initial_loss": losses[0],
        "final_loss": residual,
        "train_state_std": norm.tolist(),
        "held_out_rollout": score(model, args.traces, norm),
    }
    if args.comparison_fit:
        comparison = json.loads(args.comparison_fit.read_text())
        if comparison.get("legacy_inertias", False) != args.legacy_inertias:
            raise ValueError("The comparison fit uses a different inertia convention")
        comparison_norm = torch.tensor(comparison["train_state_std"], dtype=torch.float64)
        result["comparison_state_std"] = comparison_norm.tolist()
        result["rollout_common_normalizer"] = score(model, args.traces, comparison_norm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"held_out_rollout", "rollout_common_normalizer"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
