"""Train a Cartpole dynamics model and compare policy optimization methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from torch import nn

import newton

DT = 1.0 / 60.0
SUBSTEPS = 5
FORCE_N = 1500.0
POLICY_ACTION_LIMIT = 0.15
ASSET_SHA256 = "f31276531c03050a3ac3a3e24443899f38a4424586db17140978383433dec237"
HORIZONS = (1, 15, 100, 200)


@wp.kernel
def assign_cart_force(action: wp.array[float], joint_f: wp.array[float]):
    """Convert normalized actions to newtons at the translational joint."""
    i = wp.tid()
    joint_f[2 * i] = 1500.0 * action[i]
    joint_f[2 * i + 1] = 0.0


class NewtonCartpole:
    """Use five Featherstone substeps per recorded step in a batch of worlds."""

    def __init__(self, asset: Path, n: int, device: torch.device, *, legacy_inertias: bool = False):
        if hashlib.sha256(asset.read_bytes()).hexdigest() != ASSET_SHA256:
            raise ValueError("The Cartpole URDF differs from the pinned public artifact")
        self.n = n
        self.device = device
        body = newton.ModelBuilder()
        body.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.01, limit_ke=1.0e4, limit_kd=1.0e1)
        body.add_urdf(
            str(asset),
            xform=wp.transform(
                p=wp.vec3(0.0),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0),
            ),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=legacy_inertias,
        )
        if body.joint_coord_count != 2 or body.joint_dof_count != 2:
            raise ValueError("The benchmark requires two generalized coordinates")
        scene = newton.ModelBuilder()
        scene.replicate(body, n)
        self.model = scene.finalize(device=str(device))
        self.solver = newton.solvers.SolverFeatherstone(self.model, update_mass_matrix_interval=SUBSTEPS)
        self.current = self.model.state()
        self.next = self.model.state()
        self.control = self.model.control()
        self.action = torch.zeros(n, device=device)
        self.action_warp = wp.from_torch(self.action)
        self.advance_physics()
        self.graph = None
        if device.type == "cuda":
            with wp.ScopedCapture(device=str(device)) as capture:
                self.advance_physics()
            self.graph = capture.graph

    def reset(self, state: torch.Tensor) -> torch.Tensor:
        wp.to_torch(self.current.joint_q).view(self.n, 2).copy_(state[:, :2])
        wp.to_torch(self.current.joint_qd).view(self.n, 2).copy_(state[:, 2:])
        newton.eval_fk(self.model, self.current.joint_q, self.current.joint_qd, self.current)
        return state

    def advance_physics(self) -> None:
        wp.launch(
            assign_cart_force,
            dim=self.n,
            inputs=[self.action_warp, self.control.joint_f],
            device=str(self.device),
        )
        for _ in range(SUBSTEPS):
            self.current.clear_forces()
            self.solver.step(self.current, self.next, self.control, None, DT / SUBSTEPS)
            self.current, self.next = self.next, self.current
        # Odd substep count swaps the state buffers. Keep a fixed graph boundary.
        self.next.assign(self.current)
        self.current, self.next = self.next, self.current

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        action = action.reshape(self.n)
        self.action.copy_(action)
        if self.graph is None:
            self.advance_physics()
        else:
            wp.capture_launch(self.graph)
        q = wp.to_torch(self.current.joint_q).view(self.n, 2)
        v = wp.to_torch(self.current.joint_qd).view(self.n, 2)
        return torch.cat((q, v), dim=1)


def random_state(rng: np.random.Generator, n: int, scales: list[float], device: torch.device) -> torch.Tensor:
    state = rng.uniform(-1.0, 1.0, (n, 4)) * scales
    return torch.tensor(state, device=device, dtype=torch.float32)


def controls(rng: np.random.Generator, cases: int, horizon: int, device: torch.device) -> torch.Tensor:
    # A mixture of sustained forces and small jitter exposes feedback-relevant states.
    bias = rng.uniform(-0.08, 0.08, (horizon // 6 + 1, cases)).repeat(6, axis=0)
    jitter = rng.uniform(-0.025, 0.025, (horizon, cases))
    return torch.tensor(bias[:horizon] + jitter, device=device, dtype=torch.float32)


@torch.no_grad()
def teacher_trace(teacher: NewtonCartpole, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    teacher.reset(initial)
    return torch.stack([initial, *(teacher(action) for action in actions)])


def make_benchmark_traces(
    asset: Path, output: Path, device: torch.device, cases: int, *, legacy_inertias: bool = False
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    teacher = NewtonCartpole(asset, cases, device, legacy_inertias=legacy_inertias)
    rng = np.random.default_rng(500)
    passive_initial = random_state(rng, cases, [1.0, math.pi, 1.0, 1.0], device)
    active_initial = random_state(rng, cases, [0.8, 0.4, 0.7, 1.0], device)
    passive_actions = torch.zeros(210, cases, device=device)
    active_actions = controls(rng, cases, 210, device)
    np.savez_compressed(
        output / "shared-cartpole-traces.npz",
        legacy_inertias=np.asarray(legacy_inertias),
        passive=teacher_trace(teacher, passive_initial, passive_actions).cpu().numpy(),
        passive_actions=passive_actions.cpu().numpy(),
        active=teacher_trace(teacher, active_initial, active_actions).cpu().numpy(),
        active_actions=active_actions.cpu().numpy(),
    )


def elapsed_since(start: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


class Dynamics(nn.Module):
    """Predict a small residual over kinematic integration in joint space."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state_drift_scale", torch.tensor([0.005, 0.008, 0.5, 0.8]))
        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, theta, xdot, omega = state.unbind(dim=-1)
        features = torch.stack(
            (
                x * 0.25,
                theta.sin(),
                theta.cos(),
                xdot * 0.2,
                omega * 0.1,
                action.reshape_as(x) / POLICY_ACTION_LIMIT,
                (x.abs() - 3.7).clamp_min(0.0) * x.sign(),
            ),
            dim=-1,
        )
        correction = self.network(features) * self.state_drift_scale
        return state + torch.cat((DT * state[..., 2:], torch.zeros_like(state[..., 2:])), dim=-1) + correction


def fit_model(
    asset: Path,
    output: Path,
    device: torch.device,
    n: int,
    steps: int,
    *,
    legacy_inertias: bool = False,
) -> tuple[Dynamics, dict]:
    generator = torch.Generator(device=device).manual_seed(2)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    teacher = NewtonCartpole(asset, n, device, legacy_inertias=legacy_inertias)
    setup_s = elapsed_since(start, device)
    start = time.perf_counter()
    s0 = random_state(rng, n, [3.85, math.pi, 6.0, 8.0], device)
    s0[n // 2 :] = random_state(rng, n - n // 2, [0.8, 0.5, 1.0, 1.2], device)
    actions = controls(rng, n, 32, device)
    actions[:4, : n // 4] = torch.tensor(
        rng.uniform(-POLICY_ACTION_LIMIT, POLICY_ACTION_LIMIT, (4, n // 4)),
        device=device,
        dtype=torch.float32,
    )
    states = teacher_trace(teacher, s0, actions)
    data_s = elapsed_since(start, device)
    torch.manual_seed(3)
    model = Dynamics().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, fused=(device.type == "cuda"))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)
    start = time.perf_counter()
    rows = []
    for update in range(steps):
        # Large one-step batches first, then short closed-loop rollout losses.
        horizon = 1 if update < int(steps * 0.8) else 4
        t = torch.randint(actions.shape[0] - horizon + 1, (1024,), generator=generator, device=device)
        b = torch.randint(n, (1024,), generator=generator, device=device)
        predicted = states[t, b]
        loss = torch.zeros((), device=device)
        for k in range(horizon):
            predicted = model(predicted, actions[t + k, b])
            scale = model.state_drift_scale * (k + 1)
            loss = loss + ((predicted - states[t + k + 1, b]) / scale).square().mean()
        opt.zero_grad(set_to_none=True)
        (loss / horizon).backward()
        opt.step()
        schedule.step()
        if (update + 1) % max(1, steps // 8) == 0 or update == 0:
            rows.append({"update": update + 1, "loss": float(loss.detach() / horizon), "horizon": horizon})
    fit_s = elapsed_since(start, device)
    torch.save(model.state_dict(), output / "model.pt")
    metrics = {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "legacy_inertias": legacy_inertias,
        "train_transitions": int(actions.numel()),
        "train_state_std": states.std(dim=(0, 1)).tolist(),
        "teacher_setup_s": setup_s,
        "teacher_rollout_s": data_s,
        "fit_s": fit_s,
        "train_curve": rows,
    }
    model.eval().requires_grad_(False)
    return model, metrics


def wrapped_error(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    error = prediction - reference
    theta_error = error[..., 1]
    return torch.stack(
        (error[..., 0], torch.atan2(theta_error.sin(), theta_error.cos()), error[..., 2], error[..., 3]), dim=-1
    )


@torch.no_grad()
def evaluate_model(model: Dynamics, traces: Path, device: torch.device, output: Path, train_std: list[float]) -> dict:
    result = {}
    paired = {}
    fit_normalization = torch.tensor(train_std, device=device)
    with np.load(traces) as source:
        for scenario in ("passive", "active"):
            reference = torch.tensor(source[scenario][10:], device=device)
            actions = torch.tensor(source[f"{scenario}_actions"][10:], device=device)
            predicted = reference[0]
            selected = [predicted]
            scenario_result = {}
            normalization = reference.std(dim=(0, 1))
            prefix_sse = torch.zeros(4, device=device)
            for k, action in enumerate(actions, 1):
                predicted = model(predicted, action)
                selected.append(predicted)
                error = wrapped_error(predicted, reference[k])
                frame_mse = error.square().mean(dim=0)
                prefix_sse = prefix_sse + frame_mse
                if k in HORIZONS:
                    rmse = frame_mse.sqrt()
                    prefix_rmse = (prefix_sse / k).sqrt()
                    scenario_result[str(k)] = {
                        "endpoint_rmse_by_state": rmse.tolist(),
                        "endpoint_nrmse": float((rmse / normalization).square().mean().sqrt()),
                        "reference_std_by_state": normalization.tolist(),
                        "prefix_rmse_by_state": prefix_rmse.tolist(),
                        "prefix_nrmse": float((prefix_rmse / fit_normalization).square().mean().sqrt()),
                    }
            prediction = torch.stack(selected)
            paired[f"{scenario}_prediction"] = prediction[:, :16].cpu().numpy()
            paired[f"{scenario}_teacher"] = reference[:, :16].cpu().numpy()
            paired[f"{scenario}_actions"] = actions[:, :16].cpu().numpy()
            one_step_prediction = model(reference[:-1], actions)
            one_step_rmse = wrapped_error(one_step_prediction, reference[1:]).square().mean(dim=(0, 1)).sqrt()
            scenario_result["teacher_forced_rmse_by_state"] = one_step_rmse.tolist()
            result[scenario] = scenario_result
    np.savez_compressed(output / "cartpole-rollout-visual.npz", **paired)
    return result


class Actor(nn.Module):
    """Map scaled joint observations to the mean of a bounded action."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.uniform_(self.layers[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(policy_features(state)).squeeze(-1)


def policy_features(state: torch.Tensor) -> torch.Tensor:
    x, theta, xdot, omega = state.unbind(dim=-1)
    return torch.stack((x, torch.atan2(theta.sin(), theta.cos()) * 2.0, xdot * 0.4, omega * 0.4), dim=-1)


def action_from_logit(logit: torch.Tensor) -> torch.Tensor:
    return POLICY_ACTION_LIMIT * logit.tanh()


def stage_cost(state: torch.Tensor) -> torch.Tensor:
    x, theta, xdot, omega = state.unbind(dim=-1)
    angle = torch.atan2(theta.sin(), theta.cos())
    return angle.square() + 0.05 * x.square() + 0.1 * (xdot.square() + omega.square())


def policy_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # The shared surrogate objective adds gentle constraint costs without hard termination.
    x, _, xdot, omega = state.unbind(dim=-1)
    violation = (
        (x.abs() - 4.0).clamp_min(0.0).square()
        + (xdot.abs() - 10.0).clamp_min(0.0).square()
        + (omega.abs() - 10.0).clamp_min(0.0).square()
    )
    return stage_cost(state) + 0.05 * (action / POLICY_ACTION_LIMIT).square() + 2.0 * violation


@torch.no_grad()
def evaluate_policy(
    actor: Actor,
    teacher: NewtonCartpole,
    initial: torch.Tensor,
    horizon: int,
    *,
    save_trace: bool = False,
) -> tuple[dict, dict]:
    state = teacher.reset(initial)
    held_count = torch.zeros(len(initial), dtype=torch.long, device=initial.device)
    safe = torch.ones_like(held_count, dtype=torch.bool)
    cost = torch.zeros(len(initial), device=initial.device)
    records = [initial] if save_trace else []
    force_history = []
    for k in range(horizon):
        action = action_from_logit(actor(state))
        state = teacher(action)
        x, theta, xdot, omega = state.unbind(dim=-1)
        angle = torch.atan2(theta.sin(), theta.cos())
        safe &= (x.abs() <= 4.0) & (xdot.abs() <= 10.0) & (omega.abs() <= 10.0)
        if k >= horizon - 60:
            held_count += (angle.abs() <= 0.15) & (omega.abs() <= 0.5)
        cost += stage_cost(state)
        if save_trace:
            records.append(state)
            force_history.append(action)
    valid_success = safe & (held_count == 60)
    metrics = {
        "mean_native_cost": float((cost / horizon).mean()),
        "median_native_cost": float((cost / horizon).median()),
        "success_rate": float(valid_success.float().mean()),
        "safe_rate": float(safe.float().mean()),
        "cases": len(initial),
        "horizon": horizon,
    }
    trace = (
        {
            "state": torch.stack(records)[:, :24].cpu().numpy(),
            "action": torch.stack(force_history)[:, :24].cpu().numpy(),
        }
        if save_trace
        else {}
    )
    return metrics, trace


PPO_STD = 0.18


def log_likelihood(values: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((values - means) / PPO_STD).square()


@torch.no_grad()
def sample_episodes(
    actor: Actor,
    critic: nn.Module,
    dynamics: Dynamics | NewtonCartpole,
    initial: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if isinstance(dynamics, NewtonCartpole):
        dynamics.reset(initial)
    state = initial
    states, logits, log_probs, values, costs = [], [], [], [], []
    for jitter in noise:
        mean = actor(state)
        sampled = mean + PPO_STD * jitter
        action = action_from_logit(sampled)
        states.append(state)
        logits.append(sampled)
        log_probs.append(log_likelihood(sampled, mean))
        values.append(critic(policy_features(state)).squeeze(-1))
        state = dynamics(action) if isinstance(dynamics, NewtonCartpole) else dynamics(state, action)
        costs.append(policy_cost(state, action))
    # A fixed-horizon undiscounted objective is shared with pathwise differentiation.
    gae = torch.zeros_like(values[-1])
    next_value = torch.zeros_like(values[-1])
    advantages = []
    for value, cost in zip(reversed(values), reversed(costs)):
        gae = -cost + next_value - value + 0.95 * gae
        advantages.append(gae)
        next_value = value
    advantages = torch.stack(list(reversed(advantages)))
    returns = advantages + torch.stack(values)
    flat = (
        torch.cat(states),
        torch.cat(logits),
        torch.cat(log_probs),
        advantages.flatten(),
        returns.flatten(),
    )
    return flat


def ppo_update(
    actor: Actor,
    critic: nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    data: tuple[torch.Tensor, ...],
    generator: torch.Generator,
) -> None:
    states, logits, log_prob, advantage, returns = data
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    for _ in range(4):
        order = torch.randperm(len(states), device=states.device, generator=generator)
        for indices in order.split(1024):
            obs = states[indices]
            ratio = (log_likelihood(logits[indices], actor(obs)) - log_prob[indices]).clamp(-12, 12).exp()
            unclipped = ratio * advantage[indices]
            clipped = ratio.clamp(0.8, 1.2) * advantage[indices]
            actor_opt.zero_grad(set_to_none=True)
            (-torch.minimum(unclipped, clipped).mean()).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
            critic_opt.zero_grad(set_to_none=True)
            (critic(policy_features(obs)).squeeze(-1) - returns[indices]).square().mean().backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()


def pathwise_update(
    actor: Actor,
    dynamics: Dynamics,
    optimizer: torch.optim.Optimizer,
    initial: torch.Tensor,
    noise: torch.Tensor,
) -> None:
    state = initial
    costs = []
    for jitter in noise:
        action = action_from_logit(actor(state) + PPO_STD * jitter)
        state = dynamics(state, action)
        costs.append(policy_cost(state, action))
    loss = torch.stack(costs).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()


def evaluate_optimizers(
    model: Dynamics,
    asset: Path,
    output: Path,
    device: torch.device,
    updates: int,
    seeds: tuple[int, ...],
    world_count: int,
    holdout_count: int,
    horizon: int,
    methods: tuple[str, ...],
    *,
    legacy_inertias: bool = False,
) -> dict:
    eval_teacher = NewtonCartpole(asset, holdout_count, device, legacy_inertias=legacy_inertias)
    held = random_state(np.random.default_rng(1729), holdout_count, [0.5, 0.25, 0.5, 0.5], device)
    true_dynamics = (
        NewtonCartpole(asset, world_count, device, legacy_inertias=legacy_inertias) if "newton_ppo" in methods else None
    )
    result = {
        "world_count": world_count,
        "train_horizon": horizon,
        "eval_horizon": 120,
        "eval_cases": holdout_count,
        "success": "No native termination and |theta| <= .15 rad, |omega| <= .5 rad/s for final 60 frames",
        "arms": {},
    }
    traces = {}
    for method in methods:
        runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            actor = Actor().to(device)
            critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
            actor_opt = torch.optim.Adam(
                actor.parameters(), lr=0.010 if method == "pathwise" else 0.001, fused=(device.type == "cuda")
            )
            critic_opt = torch.optim.Adam(critic.parameters(), lr=0.002, fused=(device.type == "cuda"))
            start_state_rng = np.random.default_rng(seed)
            exploration_rng = torch.Generator(device=device).manual_seed(seed + 31)
            minibatch_rng = torch.Generator(device=device).manual_seed(seed + 71)
            training_s = 0.0
            curve = []
            for update in range(updates + 1):
                if update % max(1, min(8, updates // 4)) == 0 or update == updates:
                    eval_start = time.perf_counter()
                    stats, trace = evaluate_policy(actor, eval_teacher, held, 120, save_trace=(update == updates))
                    curve.append(
                        {
                            "update": update,
                            "simulator_steps": update * horizon * world_count,
                            "training_s": training_s,
                            "evaluation_s": elapsed_since(eval_start, device),
                            "newton": stats,
                        }
                    )
                    if update == updates:
                        traces.update({f"{method}_{seed}_{key}": value for key, value in trace.items()})
                        torch.save(actor.state_dict(), output / f"actor-{method}-{seed}.pt")
                if update == updates:
                    break
                start = time.perf_counter()
                initial = random_state(start_state_rng, world_count, [0.5, 0.25, 0.5, 0.5], device)
                noise = torch.randn((horizon, world_count), device=device, generator=exploration_rng)
                if method == "pathwise":
                    pathwise_update(actor, model, actor_opt, initial, noise)
                else:
                    dynamics = true_dynamics if method == "newton_ppo" else model
                    data = sample_episodes(actor, critic, dynamics, initial, noise)
                    ppo_update(actor, critic, actor_opt, critic_opt, data, minibatch_rng)
                training_s += elapsed_since(start, device)
            runs.append({"seed": seed, "curve": curve})
            print(f"{method} seed {seed}: {curve[-1]}", flush=True)
        result["arms"][method] = {"seeds": runs}
        (output / "policy-progress.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "cartpole-policy-visual.npz", **traces)
    return result


@torch.no_grad()
def timing(model: Dynamics, asset: Path, device: torch.device, *, legacy_inertias: bool = False) -> dict:
    timing_result = {}
    for batch in (1, 128, 2048):
        teacher = NewtonCartpole(asset, batch, device, legacy_inertias=legacy_inertias)
        teacher_zeros = torch.zeros((batch, 4), device=device)
        zeros_action = teacher_zeros[:, 0].contiguous()
        teacher.reset(teacher_zeros)
        model_state = teacher_zeros.clone()
        for _ in range(8):
            model_state = model(model_state, zeros_action)
        graph = None
        if device.type == "cuda":
            graph = torch.cuda.CUDAGraph()
            old = torch.cuda.current_stream(device)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(old)
            with torch.cuda.graph(graph, stream=capture_stream):
                model_state.copy_(model(model_state, zeros_action))
            old.wait_stream(capture_stream)

        def evaluate_actor(name: str) -> None:
            if name == "newton":
                teacher(zeros_action)
            elif name == "newton_graph":
                wp.capture_launch(teacher.graph)
            elif name == "learned_eager":
                model_state.copy_(model(model_state, zeros_action))
            else:
                graph.replay()

        measured = {}
        names = ["newton", "learned_eager"] + (["newton_graph", "learned_graph"] if graph else [])
        for name in names:
            for _ in range(10):
                evaluate_actor(name)
            elapsed_since(time.perf_counter(), device)
            individual = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(100):
                    evaluate_actor(name)
                individual.append(10.0 * elapsed_since(t0, device))
            measured[name] = {"ms_p50": float(np.median(individual)), "ms_samples": individual}
        timing_result[str(batch)] = measured
    return timing_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("run"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--evaluation-cases", type=int, default=2048)
    parser.add_argument("--train-worlds", type=int, default=4096)
    parser.add_argument("--fit-updates", type=int, default=2200)
    parser.add_argument("--policy-updates", type=int, default=96)
    parser.add_argument("--policy-worlds", type=int, default=64)
    parser.add_argument("--policy-eval-cases", type=int, default=256)
    parser.add_argument("--policy-horizon", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=(5, 19, 47))
    parser.add_argument("--traces-only", action="store_true")
    parser.add_argument(
        "--legacy-inertias",
        action="store_true",
        help="Infer shape masses at 1000 kg/m^3 as the legacy Warp example did",
    )
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--reuse-fit", action="store_true")
    parser.add_argument("--newton-ppo", action="store_true")
    parser.add_argument("--policy-methods", choices=["pathwise", "ppo", "newton_ppo"], nargs="+")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(str(device))))
    start = time.perf_counter()
    if not (args.output / "shared-cartpole-traces.npz").exists():
        make_benchmark_traces(
            args.urdf, args.output, device, args.evaluation_cases, legacy_inertias=args.legacy_inertias
        )
    with np.load(args.output / "shared-cartpole-traces.npz") as held_out:
        if bool(held_out.get("legacy_inertias", False)) != args.legacy_inertias:
            raise ValueError("The cached traces use a different Cartpole inertia protocol")
        if held_out["passive"].shape[1] != args.evaluation_cases:
            raise ValueError("The cached traces use a different evaluation case count")
    eval_data_s = elapsed_since(start, device)
    if args.traces_only:
        return
    fit_receipt = args.output / "fit.json"
    methods = args.policy_methods or (["pathwise", "ppo", "newton_ppo"] if args.newton_ppo else ["pathwise", "ppo"])
    if args.reuse_fit:
        model = Dynamics().to(device)
        model.load_state_dict(torch.load(args.output / "model.pt", map_location=device, weights_only=True))
        model.eval().requires_grad_(False)
        model_metrics = json.loads(fit_receipt.read_text())
        if model_metrics.get("legacy_inertias", False) != args.legacy_inertias:
            raise ValueError("The cached checkpoint uses a different Cartpole inertia protocol")
    else:
        model, model_metrics = fit_model(
            args.urdf, args.output, device, args.train_worlds, args.fit_updates, legacy_inertias=args.legacy_inertias
        )
        model_metrics["rollout"] = evaluate_model(
            model,
            args.output / "shared-cartpole-traces.npz",
            device,
            args.output,
            model_metrics["train_state_std"],
        )
        fit_receipt.write_text(json.dumps(model_metrics, indent=2) + "\n")
        print(f"Model fit: {model_metrics}", flush=True)
    if args.fit_only:
        return
    result = {
        "protocol": {
            "version": 1,
            "physics_hz": 1.0 / DT,
            "substeps": SUBSTEPS,
            "state_order": ["cart_m", "pole_rad", "cart_m_s", "pole_rad_s"],
            "solver": "Newton SolverFeatherstone",
            "asset_sha256": ASSET_SHA256,
            "policy_force_limit_n": FORCE_N * POLICY_ACTION_LIMIT,
            "legacy_inertias": args.legacy_inertias,
            "model_test_seed": 500,
            "policy_test_seed": 1729,
        },
        "run": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "newton": newton.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "eval_data_s_this_invocation": eval_data_s,
            "argv": {k: str(v) for k, v in vars(args).items()},
        },
        "model": model_metrics,
        "timing": {} if args.skip_timing else timing(model, args.urdf, device, legacy_inertias=args.legacy_inertias),
    }
    if args.timing_only:
        (args.output / "timing.json").write_text(json.dumps(result["timing"], indent=2) + "\n")
        return
    receipt = args.output / "results.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    result["policy"] = evaluate_optimizers(
        model,
        args.urdf,
        args.output,
        device,
        args.policy_updates,
        tuple(args.seeds),
        args.policy_worlds,
        args.policy_eval_cases,
        args.policy_horizon,
        tuple(methods),
        legacy_inertias=args.legacy_inertias,
    )
    receipt.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
