"""Compare legacy Warp Cartpole with the current Newton same-inertia traces."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from official_nerd_replay import state_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nerd-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ("git", "-C", str(args.nerd_dir), "rev-parse", "HEAD"), text=True
    ).strip()
    if revision != "370e89d2b8ce68aa3578aa7812859c681b7502bc" or wp.__version__ != "1.8.0":
        raise ValueError("Use the original NeRD revision and Warp 1.8.0.")

    sys.path.insert(0, str(args.nerd_dir.resolve()))
    from envs.neural_environment import NeuralEnvironment

    device = torch.device("cuda:0")
    torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(str(device))))
    result = {"original_nerd_commit": revision, "warp_version": wp.__version__, "cohorts": {}}
    with np.load(args.input, allow_pickle=False) as source, torch.no_grad():
        if not bool(source.get("legacy_inertias", False)):
            raise ValueError("Expected modern Newton traces recorded with --legacy-inertias.")
        env = None
        for cohort in ("passive", "active"):
            reference = torch.as_tensor(source[cohort], device=device)
            actions = torch.as_tensor(source[f"{cohort}_actions"], device=device).unsqueeze(-1)
            if env is None:
                env = NeuralEnvironment(
                    env_name="Cartpole",
                    num_envs=reference.size(1),
                    warp_env_cfg={"seed": 500},
                    default_env_mode="ground-truth",
                    render=False,
                )
            env.reset(reference[10])
            prediction = torch.stack([env.states.clone(), *(env.step(action).clone() for action in actions[10:])])
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite analytical rollout in the original Warp environment.")
            result["cohorts"][cohort] = state_metrics(
                prediction.cpu().numpy().swapaxes(0, 1), reference[10:].cpu().numpy().swapaxes(0, 1)
            )
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["frame_dt_seconds"] = env.env.frame_dt
    result["cases"] = env.num_envs
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
