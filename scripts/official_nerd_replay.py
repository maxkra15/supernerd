"""Replay the published NVlabs Cartpole checkpoint in its Newton adapter."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import warp as wp
import yaml


NERD_REVISION = "476dc67fccc6a99be293bf611f897b1c06f13e30"
CHECKPOINT_SHA = "287dde2134eb8f5277c17a8f7cc7abbe64063e97cb481bdb6e8d76f9536e5366"
STATE_FIELDS = ("cart_position_m", "pole_angle_rad", "cart_velocity_m_s", "pole_angular_velocity_rad_s")


def state_metrics(prediction, reference):
    difference = (prediction.astype(np.float64) - reference.astype(np.float64))[:, 1:]
    difference[..., 1] = np.arctan2(np.sin(difference[..., 1]), np.cos(difference[..., 1]))
    scales = reference.std(axis=(0, 1), dtype=np.float64)
    summary = {}
    for prefix in (1, 8, 15, 32, 100, 200):
        if prefix <= difference.shape[1]:
            prefix_difference = difference[:, :prefix]
            summary[str(prefix)] = {
                "prefix_rmse": dict(zip(STATE_FIELDS, np.sqrt(np.mean(prefix_difference**2, axis=(0, 1))).tolist())),
                "terminal_rmse": dict(zip(STATE_FIELDS, np.sqrt(np.mean(prefix_difference[:, -1] ** 2, axis=0)).tolist())),
                "prefix_nrmse": float(np.sqrt(np.mean((prefix_difference / scales) ** 2))),
                "terminal_nrmse": float(np.sqrt(np.mean((prefix_difference[:, -1] / scales) ** 2))),
            }
    return summary


def rollout(env, action, prime_states, prime_actions):
    env.reset(prime_states[0])
    env.init_rnn(env.num_envs)
    states = torch.empty((env.num_envs, action.size(0) + 1, env.state_dim), device=action.device)
    actual_force = torch.empty_like(action)
    with torch.inference_mode():
        for state, command in zip(prime_states[:-1], prime_actions, strict=True):
            env._update_states(state)
            env.step(command)
        env._update_states(prime_states[-1])
        states[:, 0] = env.states
        for step, command in enumerate(action):
            states[:, step + 1] = env.step(command)
            actual_force[step, :, 0] = env.joint_f[:, 0]
    return states, actual_force


def measure_throughput(make_environment, device):
    results = {}
    for batch in (1, 128, 2048):
        env = make_environment(batch)
        env.default_env_mode = "neural"
        action = torch.zeros((batch, 1), device=device)
        starting_state = torch.zeros((batch, 4), device=device)
        elapsed = []
        with torch.inference_mode():
            for _ in range(5):
                env.reset(starting_state)
                env.init_rnn(batch)
                for _ in range(10):
                    env.step(action)
                torch.cuda.synchronize(device)
                started = perf_counter()
                for _ in range(100):
                    env.step(action)
                torch.cuda.synchronize(device)
                elapsed.append(perf_counter() - started)
        results[str(batch)] = {
            "block_seconds": elapsed,
            "median_ms_per_step": 1000 * median(elapsed) / 100,
            "median_transitions_per_second": 100 * batch / median(elapsed),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nerd-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Optional NPZ with time-first {cohort} and {cohort}_actions arrays.")
    parser.add_argument("--cohort", choices=("passive", "active"), default="passive")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    if not args.timing_only and args.input is None:
        parser.error("a paired --input trace is required for accuracy evaluation")
    nerd_dir = args.nerd_dir.resolve()
    revision = subprocess.check_output(("git", "-C", str(nerd_dir), "rev-parse", "HEAD"), text=True).strip()
    if revision != NERD_REVISION:
        raise ValueError(f"Expected NVlabs NeRD commit {NERD_REVISION}, got {revision}.")
    checkpoint = nerd_dir / "pretrained_models/NeRD_models/Cartpole/model/nn/model.pt"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != CHECKPOINT_SHA:
        raise ValueError(f"Official LFS checkpoint SHA256 differs: {checkpoint_sha}.")

    sys.path.insert(0, str(nerd_dir))
    from envs.neural_environment import NeuralEnvironment
    from envs.newton_envs.environment import SolverType

    # This pinned upstream checkpoint stores a full PyTorch module.
    model, robot_name = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    model.eval()
    model.fix_input_names()
    cfg = yaml.safe_load((checkpoint.parents[1] / "cfg.yaml").read_text())["env"]["neural_solver_cfg"]
    device = torch.device("cuda:0")
    torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(str(device))))

    def make_environment(cases):
        return NeuralEnvironment(
            env_name="Cartpole",
            num_envs=cases,
            newton_env_cfg={"seed": 500, "random_reset": True, "solver_type": SolverType.FEATHERSTONE},
            neural_solver_cfg=cfg,
            neural_model=model,
            default_env_mode="ground-truth",
            use_graph_capture=False,
            render=False,
        )

    if args.timing_only:
        result = {
            "nerd_revision": revision,
            "checkpoint_sha256": checkpoint_sha,
            "device": torch.cuda.get_device_name(device),
            "warp_version": wp.__version__,
            "torch_version": torch.__version__,
            "protocol": "Eager author adapter, fixed GPU inputs, 10 warm steps and 5 groups of 100 synced steps, no policy/rendering",
            "timings": measure_throughput(make_environment, device),
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        return

    with np.load(args.input, allow_pickle=False) as inputs:
        reference = inputs[args.cohort].swapaxes(0, 1)
        commands = inputs[f"{args.cohort}_actions"].swapaxes(0, 1)
    if commands.ndim == 2:
        commands = commands[..., None]
    if reference.ndim != 3 or reference.shape[2] != 4 or commands.shape != (reference.shape[0], reference.shape[1] - 1, 1):
        raise ValueError("Expected paired states [case,time+1,4] and commands [case,time,1].")
    if not np.isfinite(reference).all() or not np.isfinite(commands).all() or np.abs(commands).max() > 1:
        raise ValueError("States must be finite and normalized actions must be in [-1,1].")
    cases = reference.shape[0]
    if not 0 <= args.warmup < commands.shape[1]:
        raise ValueError("Warmup must precede the final command.")
    prime_states = torch.as_tensor(reference[:, : args.warmup + 1].swapaxes(0, 1).copy(), dtype=torch.float32, device=device)
    prime_actions = torch.as_tensor(commands[:, : args.warmup].swapaxes(0, 1).copy(), dtype=torch.float32, device=device)
    reference = reference[:, args.warmup :]
    action = torch.as_tensor(commands[:, args.warmup :].swapaxes(0, 1).copy(), dtype=torch.float32, device=device)

    env = make_environment(cases)
    env.default_env_mode = "neural"
    prediction, force_neural = rollout(env, action, prime_states, prime_actions)
    force_delta_max = float(torch.abs(action * env.control_gains[0] - force_neural).max())
    if force_delta_max > 1e-6:
        raise ValueError(f"Replay actions produce different applied forces: {force_delta_max} N.")
    prediction = prediction.cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite state in the checkpoint rollout.")
    result = {
        "nerd_revision": revision,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_robot_name": robot_name,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "warp_version": wp.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "cases": cases,
        "horizon": action.size(0),
        "frame_dt_seconds": env.frame_dt,
        "action_gain_newtons": env.control_gains[0],
        "max_applied_force_disagreement_newtons": force_delta_max,
        "history_initialization": f"{args.warmup} teacher steps followed by recursive rollout",
        "prefix_and_terminal_error": state_metrics(prediction, reference),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_cohort": args.cohort,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_npz:
        np.savez_compressed(args.output_npz, analytical=reference, prediction=prediction)
    print(args.result)


if __name__ == "__main__":
    main()
