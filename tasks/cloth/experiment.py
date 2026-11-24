"""Paired source-only tabletop fold benchmark on held asset/target pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from folding import (
    HOLD_HEIGHT,
    action,
    capture,
    geometry,
)

TRAIN = 192
TEST = 16
N_ACTIONS = 12


def task(i, base_seed=50100):
    seed = base_seed + i
    geometry(seed)
    landing = float(np.random.default_rng(seed + 1017).uniform(0.03, 0.33))
    controls = [[1.0, 0.0, HOLD_HEIGHT], [landing, 0.45, 0.025]]
    # Script plus a fixed, locally varying bank. Every planner receives these same twelve paths.
    for offset, lift, height in [
        (-0.06, 0.28, 0.02),
        (-0.06, 0.48, 0.055),
        (-0.03, 0.54, 0.025),
        (-0.03, 0.32, 0.045),
        (0, 0.26, 0.025),
        (0, 0.57, 0.06),
        (0.03, 0.33, 0.025),
        (0.03, 0.55, 0.04),
        (0.06, 0.28, 0.04),
        (0.06, 0.48, 0.025),
    ]:
        controls.append([landing + offset, lift, height])
    return seed, landing, np.array(controls, dtype=np.float32)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("which", choices=["train", "test", "confirm"])
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    if args.which == "train":
        jobs = [(f"train-{i:03}", 22000 + i, action(22000 + i)) for i in range(TRAIN)]
        # Eight additional no-movement examples make the no-action option in-distribution.
        jobs += [
            (f"noop-{i:03}", 28000 + i, np.array([1.0, 0.0, HOLD_HEIGHT]))
            for i in range(8)
        ]
    else:
        jobs = []
        for i in range(TEST):
            seed, _landing, controls = task(
                i, 50300 if args.which == "confirm" else 50100
            )
            jobs.extend(
                (f"test-{i:02}-{j:02}", seed, a) for j, a in enumerate(controls)
            )
    elapsed = capture(jobs, args.output, args.workers)
    (args.output / "capture.json").write_text(
        json.dumps({"episodes": len(jobs), "workers": args.workers, "seconds": elapsed})
        + "\n"
    )
    print(args.which, "complete", elapsed, flush=True)


if __name__ == "__main__":
    main()
