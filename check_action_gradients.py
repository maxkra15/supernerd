"""Compare learned CartPole action derivatives with Newton finite differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from benchmark import NewtonCartpole, teacher_trace

HORIZONS = (1, 8, 32)
CASES_PER_COHORT = 16
SEED = 71492
FD_STEPS = (0.0005, 0.00025)


def frame_cost(state: torch.Tensor) -> torch.Tensor:
    x, angle, vx, omega = state.unbind(dim=-1)
    return x.square() * 0.1 + 1 - angle.cos() + 0.02 * (vx.square() + omega.square())


def make_cases(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(SEED)
    scales = ([0.45, 0.24, 0.5, 0.85], [1.2, 2.75, 1.5, 3.0])
    initial = np.concatenate([rng.uniform(-1, 1, (CASES_PER_COHORT, 4)) * s for s in scales])
    actions = rng.uniform(-0.025, 0.025, (8, len(initial))).repeat(4, axis=0)
    return (
        torch.tensor(initial, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
    )


def teacher_jacobians(
    teacher: NewtonCartpole, initial: torch.Tensor, actions: torch.Tensor
) -> dict[float, dict[int, torch.Tensor]]:
    length, cases = actions.shape
    basis = torch.eye(length, device=actions.device)[:, None, :, None]
    signs = torch.tensor([-1.0, 1.0], device=actions.device)[None, None, None, :]
    repeated = initial[:, None, None, :].expand(cases, length, 2, 4).reshape(-1, 4).contiguous()
    results = {}
    for epsilon in FD_STEPS:
        perturbed = actions[:, :, None, None] + epsilon * basis * signs
        trace = teacher_trace(teacher, repeated, perturbed.reshape(length, -1))
        values = frame_cost(trace[1:].double()).reshape(length, cases, length, 2)
        cumulative = values.cumsum(dim=0)
        results[epsilon] = {
            h: (cumulative[h - 1, :, :h, 1] - cumulative[h - 1, :, :h, 0]) / (2 * epsilon * h)
            for h in HORIZONS
        }
    return results


def learned_cost(model: torch.nn.Module, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    state = initial
    costs = []
    for action in actions:
        state = model(state, action)
        costs.append(frame_cost(state))
    return torch.stack(costs).mean(dim=0)


def pack_teacher(
    teacher: NewtonCartpole, initial: torch.Tensor, controls: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    length, cases = controls[0].shape
    repeated = initial.repeat(len(controls), 1)
    all_controls = torch.cat(controls, dim=1)
    extra = teacher.n - len(controls) * cases
    if extra:
        repeated = torch.cat((repeated, initial[:1].expand(extra, -1)))
        all_controls = torch.cat((all_controls, all_controls[:, :1].expand(-1, extra)), dim=1)
    values = frame_cost(teacher_trace(teacher, repeated, all_controls)[1:].double()).mean(dim=0)
    return values[: len(controls) * cases].split(cases)


def evaluate(
    model: torch.nn.Module, teacher: NewtonCartpole, initial: torch.Tensor, actions: torch.Tensor
) -> dict:
    finite_differences = teacher_jacobians(teacher, initial, actions)
    rows = {}
    rng = torch.Generator(device=initial.device).manual_seed(SEED)
    for horizon in HORIZONS:
        original = actions[:horizon]
        differentiated = original.double().detach().requires_grad_()
        grad = torch.autograd.grad(learned_cost(model, initial.double(), differentiated).sum(), differentiated)[0].T
        exact = finite_differences[FD_STEPS[-1]][horizon]
        alternate = finite_differences[FD_STEPS[0]][horizon]
        scale = exact.norm(dim=1).clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(grad, exact, dim=1)
        relative = (grad - exact).norm(dim=1) / scale
        fd_sensitivity = (exact - alternate).norm(dim=1) / scale
        direction = torch.randn(grad.shape, generator=rng, device=grad.device, dtype=grad.dtype)
        direction = torch.nn.functional.normalize(direction, dim=1)
        epsilon = 1e-5
        with torch.no_grad():
            perturbed = differentiated + epsilon * direction.T
            plus = learned_cost(model, initial.double(), perturbed)
            minus = learned_cost(model, initial.double(), differentiated - epsilon * direction.T)
            torch_fd = (plus - minus) / (2 * epsilon)
        torch_fd_error = (torch_fd - (grad * direction).sum(dim=1)).abs()
        adjustment = 0.0025 * torch.nn.functional.normalize(grad, dim=1).T
        old_cost, new_cost = pack_teacher(teacher, initial, [original, original - adjustment.float()])
        improvement = (old_cost - new_cost) / old_cost.clamp_min(1e-12)
        rows[str(horizon)] = {}
        for label, selection in (
            ("near_upright", slice(0, CASES_PER_COHORT)),
            ("wide_state", slice(CASES_PER_COHORT, 2 * CASES_PER_COHORT)),
        ):
            observed = improvement[selection]
            rows[str(horizon)][label] = {
                "cases": len(observed),
                "median_cosine": cosine[selection].median().item(),
                "median_relative_gradient_error": relative[selection].median().item(),
                "median_relative_finite_difference_sensitivity": fd_sensitivity[selection].median().item(),
                "max_torch_directional_fd_error": torch_fd_error[selection].max().item(),
                "teacher_descent_count": (observed > 0).sum().item(),
                "median_teacher_relative_cost_reduction": observed.median().item(),
                "case_gradient_cosine": cosine[selection].tolist(),
                "case_teacher_relative_cost_reduction": observed.tolist(),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--architecture", choices=("mlp", "structured"), default="mlp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    from benchmark import Dynamics
    from structured_dynamics import StructuredDynamics

    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(str(device))))
    architecture = Dynamics if args.architecture == "mlp" else StructuredDynamics
    model = architecture().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model = model.eval().double()
    initial, actions = make_cases(device)
    teacher = NewtonCartpole(args.urdf, 2 * len(initial) * len(actions), device)
    results = {
        "seed": SEED,
        "architecture": args.architecture,
        "cases_per_cohort": CASES_PER_COHORT,
        "steps": list(HORIZONS),
        "newton_central_difference_action_epsilon": list(FD_STEPS),
        "tested_action_update_l2": 0.0025,
        "results": evaluate(model, teacher, initial, actions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
