"""Rank three supported fold programs; physically evaluate every candidate outside the training envelope."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from experiment import TEST, task
from folding import HOLD_HEIGHT, geometry, metrics, teacher
from learn import CENTER, SCALES, FoldResponse, plan, score

FABRIC = (
    "length_m",
    "width_m",
    "membrane_stiffness",
    "bend_stiffness",
    "areal_density",
)
COMMAND = ("landing_fraction", "lift_fraction", "release_height_m")
NO_MOVE = np.array([1.0, 0.0, HOLD_HEIGHT], dtype=np.float32)
HERE = Path(__file__).resolve().parent


class GuardedFoldPlanner:
    """Use native_score(index) to evaluate an actionable plan on the caller's own teacher."""

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self._model = None
        with np.load(self.checkpoint) as archive:
            centers = archive["centers"]
        if centers.ndim != 2 or centers.shape[1] != 8 or not np.isfinite(centers).all():
            raise ValueError(
                "Expected 8 normalized, finite training descriptors per center"
            )
        no_move_z = (NO_MOVE - CENTER[5:]) / SCALES[5:]
        # Training used both float32 and float64 raw commands before storing float32 centers.
        stationary = np.isclose(centers[:, 5:], no_move_z, rtol=0, atol=1e-6).all(
            axis=1
        )
        if stationary.all() or not stationary.any():
            raise ValueError(
                "Checkpoint must contain moving and no-motion training programs"
            )
        self.low = np.r_[centers[:, :5].min(0), centers[~stationary, 5:].min(0)]
        self.high = np.r_[centers[:, :5].max(0), centers[~stationary, 5:].max(0)]
        self.center_count = len(centers)
        self.no_move_count = int(stationary.sum())

    def calibration(self):
        """Exact applied normalized bounds plus their human-readable physical units."""
        return {
            "checkpoint_sha256": hashlib.sha256(
                self.checkpoint.read_bytes()
            ).hexdigest(),
            "training_centers": self.center_count,
            "no_motion_centers": self.no_move_count,
            "features": [*FABRIC, *COMMAND],
            "physical_min": (self.low * SCALES + CENTER).tolist(),
            "physical_max": (self.high * SCALES + CENTER).tolist(),
            "normalized_min": self.low.tolist(),
            "normalized_max": self.high.tolist(),
            "no_motion_command": NO_MOVE.tolist(),
        }

    def outside(self, g, actions):
        """List descriptors which put this fabric or at least one proposed command out of range."""
        g = np.asarray(g, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if (
            g.shape != (5,)
            or actions.ndim != 2
            or actions.shape[1] != 3
            or not len(actions)
        ):
            raise ValueError(
                "Expected one five-feature fabric and one or more three-feature commands"
            )
        if not np.isfinite(g).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite fabric or command descriptors")
        fabric = (g - CENTER[:5]) / SCALES[:5]
        moving = actions[~(actions == NO_MOVE).all(axis=1)]
        commands = (moving - CENTER[5:]) / SCALES[5:]
        outside_fabric = (fabric < self.low[:5]) | (fabric > self.high[:5])
        outside_command = ((commands < self.low[5:]) | (commands > self.high[5:])).any(
            axis=0
        )
        return [
            name
            for name, outside in zip(
                (*FABRIC, *COMMAND), np.r_[outside_fabric, outside_command], strict=True
            )
            if outside
        ]

    def select(self, g, actions, landing, native_score):
        """Return the best checked index. Every selected candidate receives one native_score call."""
        g = np.asarray(g, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if not np.isfinite(landing):
            raise ValueError("Nonfinite target landing fraction")
        outside = self.outside(g, actions)
        if outside:
            candidates = list(range(len(actions)))
        else:
            if self._model is None:
                self._model = FoldResponse(self.checkpoint).eval()
            _, _, predicted = plan(self._model, g, actions, landing)
            costs = [score(m, g) for m in predicted]
            candidates = np.argsort(costs, kind="stable")[:3].tolist()
        checked = {i: float(native_score(i)) for i in candidates}
        if not np.isfinite(list(checked.values())).all():
            raise ValueError("Native evaluator returned a nonfinite score")
        return {
            "used_model": not outside,
            "outside": outside,
            "checked": candidates,
            "native_evaluations": len(checked),
            "selected": min(checked, key=checked.__getitem__),
        }


def evaluate_seed(seed):
    """Run the planner through the production Newton cloth teacher; record its actual calls."""
    torch.set_num_threads(1)
    _, landing, actions = task(seed - 50300, 50300)
    g = geometry(seed)
    outcomes = {}

    def simulate(index):
        if index in outcomes:
            raise RuntimeError("Duplicate native evaluation")
        trajectory, _ = teacher(seed, actions[index])
        result = metrics(trajectory, g, landing)
        outcomes[index] = result
        return score(result, g)

    planner = GuardedFoldPlanner(HERE / "fold-response.npz")
    decision = planner.select(g, actions, landing, simulate)
    assert list(outcomes) == decision["checked"]
    return {
        "seed": seed,
        **decision,
        "outcome": outcomes[decision["selected"]],
        "checked_native_outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--seed", type=int, default=50300, help="Reproduce one original public fabric"
    )
    group.add_argument(
        "--confirm",
        action="store_true",
        help="Run all sixteen original confirmation cases",
    )
    group.add_argument(
        "--calibration",
        action="store_true",
        help="Print the bounds without running a teacher",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    calibration = GuardedFoldPlanner(HERE / "fold-response.npz").calibration()
    if options.calibration:
        print(json.dumps(calibration, indent=2))
        return
    seeds = list(range(50300, 50300 + TEST)) if options.confirm else [options.seed]
    if options.workers < 1:
        parser.error("--workers must be positive")
    if options.workers == 1:
        cases = [evaluate_seed(seed) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=options.workers) as pool:
            cases = list(pool.map(evaluate_seed, seeds))
    report = {
        "calibration": calibration,
        "cases": cases,
        "successes": sum(r["outcome"]["success"] for r in cases),
        "native_evaluations": sum(r["native_evaluations"] for r in cases),
        "used_model_cases": sum(r["used_model"] for r in cases),
        "mean_shape_error_percent": float(
            np.mean([r["outcome"]["shape_error_percent"] for r in cases])
        ),
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if options.output:
        options.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
