"""Train a joint-space force correction over a structured CartPole transition."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from torch import nn

from benchmark import (
    DT,
    FORCE_N,
    HORIZONS,
    SUBSTEPS,
    NewtonCartpole,
    controls,
    elapsed_since,
    random_state,
    teacher_trace,
    wrapped_error,
)


class StructuredDynamics(nn.Module):
    """Integrate CartPole with a positive-definite mass matrix and a learned force residual."""

    def __init__(self) -> None:
        super().__init__()
        self.correction = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, theta, vx, omega = state.unbind(dim=-1)
        applied = FORCE_N * action.reshape_as(x)
        # The teacher refreshes its inertia at the first substep of each decision.
        coupling = 5.0 * theta.cos()
        mass_x, inertia_theta = 11.01, 3.51
        inv_det = (mass_x * inertia_theta - coupling.square()).reciprocal()
        dt = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            sin = theta.sin()
            outside = x - x.clamp(-4.0, 4.0)
            residual_x, residual_theta = self.correction(
                torch.stack((sin, theta.cos(), vx / 5, omega / 5, applied / 200,
                             outside * 10, (outside != 0).to(x.dtype)), dim=-1)
            ).unbind(dim=-1)
            force_x = (
                applied + 5.0 * sin * omega.square() - 10000.0 * outside
                - 10.0 * vx * (outside != 0) + 10 * residual_x
            )
            force_theta = 49.05 * sin + 10 * residual_theta
            accel_x = (inertia_theta * force_x - coupling * force_theta) * inv_det
            accel_theta = (mass_x * force_theta - coupling * force_x) * inv_det
            vx = vx + dt * accel_x
            omega = omega + dt * accel_theta
            x = x + dt * vx
            theta = theta + dt * omega
        return torch.stack((x, theta, vx, omega), dim=-1)


@torch.no_grad()
def score(model: StructuredDynamics, data: Path, normalizer: torch.Tensor) -> dict:
    device = normalizer.device
    measurements = {}
    with np.load(data) as archive:
        for cohort in ("passive", "active"):
            reference = torch.tensor(archive[cohort][10:], device=device)
            actions = torch.tensor(archive[f"{cohort}_actions"][10:], device=device)
            sums = torch.zeros(4, device=device, dtype=torch.float64)
            state = reference[0]
            points = {}
            for step, action in enumerate(actions, 1):
                state = model(state, action)
                squared = wrapped_error(state, reference[step]).double().square()
                sums = sums + squared.mean(dim=0)
                if step in HORIZONS:
                    rms = (sums / step).sqrt()
                    points[str(step)] = {
                        "prefix_rmse_by_state": rms.tolist(),
                        "prefix_nrmse": (rms / normalizer).square().mean().sqrt().item(),
                        "endpoint_rmse_by_state": squared.mean(dim=0).sqrt().tolist(),
                    }
            measurements[cohort] = points
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=320)
    args = parser.parse_args()
    if min(args.worlds, args.updates) < 1:
        parser.error("world and update counts must be positive")

    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(str(device))))
    rng = np.random.default_rng(0)
    generator = torch.Generator(device=device).manual_seed(2)
    torch.manual_seed(3)

    # Match the unrestricted MLP's fit data and held-out trajectories.
    n = args.worlds
    teacher = NewtonCartpole(args.urdf, n, device)
    s0 = random_state(rng, n, [3.85, math.pi, 6.0, 8.0], device)
    s0[n // 2:] = random_state(rng, n - n // 2, [0.8, 0.5, 1.0, 1.2], device)
    actions = controls(rng, n, 32, device)
    actions[:4, :n // 4] = torch.tensor(
        rng.uniform(-0.15, 0.15, (4, n // 4)), device=device, dtype=torch.float32
    )
    states = teacher_trace(teacher, s0, actions)
    norm = states.std(dim=(0, 1)).clamp_min(1e-6)
    model = StructuredDynamics().to(device)
    analytical = score(model, args.traces, norm)

    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    scale = torch.tensor([0.005, 0.008, 0.5, 0.8], device=device)
    started = time.perf_counter()
    losses = []
    for update in range(args.updates):
        t = torch.randint(32, (1024,), generator=generator, device=device)
        b = torch.randint(n, (1024,), generator=generator, device=device)
        error = wrapped_error(model(states[t, b], actions[t, b]), states[t + 1, b])
        loss = (error / scale).square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if update == 0 or (update + 1) % max(1, args.updates // 8) == 0:
            losses.append({"update": update + 1, "loss": loss.detach().item()})

    fit_seconds = elapsed_since(started, device)
    learned = score(model.eval(), args.traces, norm)
    result = {
        "substeps": SUBSTEPS,
        "fit_transitions": int(actions.numel()),
        "updates": args.updates,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "fit_seconds": fit_seconds,
        "train_state_std": norm.tolist(),
        "losses": losses,
        "zero_residual": analytical,
        "learned_residual": learned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
