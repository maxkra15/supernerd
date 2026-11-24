"""Evaluate fixed three-shot Newton verification on an independent folding task bank."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from experiment import TEST, task
from folding import geometry, metrics, teacher
from learn import FoldResponse, contact_audit, get_path, plan, score


def get_choices(model, g, actions, landing, seed):
    primary, prediction, pred_metrics = plan(model, g, actions, landing)
    ranks = np.argsort([score(m, g) for m in pred_metrics])
    control = [
        1,
        *np.random.default_rng(seed + 3100)
        .choice(np.arange(2, len(actions)), 2, replace=False)
        .tolist(),
    ]
    return primary, ranks[:3].tolist(), control, prediction, pred_metrics


def evaluate(root):
    torch.set_num_threads(1)
    net = FoldResponse(root / "fold-response.npz").eval()
    rows = []
    for i in range(TEST):
        seed, landing, actions = task(i, 50300)
        g = geometry(seed)
        single, shortlist, randomlist, pred, pm = get_choices(
            net, g, actions, landing, seed
        )
        actual = []
        shapes = []
        for j, a in enumerate(actions):
            with np.load(get_path(root / "confirm", i, j)) as z:
                assert int(z["seed"]) == seed and np.array_equal(a, z["a"])
                q = z["q"]
            shapes.append(q)
            actual.append(metrics(q, g, landing))
        costs = [score(outcome, g) for outcome in actual]
        idx = {
            "no_move": 0,
            "script": 1,
            "neural": single,
            "script_plus_two": min(randomlist, key=costs.__getitem__),
            "neural_plus_newton": min(shortlist, key=costs.__getitem__),
            "full_newton": min(range(len(actions)), key=costs.__getitem__),
        }
        rows.append(
            {
                "seed": seed,
                "target": landing,
                "indices": idx,
                "learned_shortlist": shortlist,
                "script_shortlist": randomlist,
                "outcomes": {k: actual[j] for k, j in idx.items()},
                "predicted_neural": pm[single],
                "neural_contact": contact_audit(pred[single], g, landing),
                "native_contact": {
                    k: contact_audit(shapes[j], g, landing) for k, j in idx.items()
                },
            }
        )
        if i == 0:
            j = idx["neural_plus_newton"]
            np.savez_compressed(
                root / "hybrid-example.npz",
                q=shapes[j],
                guess=pred[j],
                script=shapes[1],
                seed=seed,
                action=actions[j],
                goal_landing_fraction=landing,
                geometry=g,
            )
    summary = {}
    for k in rows[0]["indices"]:
        results = [row["outcomes"][k] for row in rows]
        summary[k] = {
            "successes": sum(m["success"] for m in results),
            "mean_shape_error_pct": float(
                np.mean([m["shape_error_percent"] for m in results])
            ),
            "final_intersecting_cases": sum(
                row["native_contact"][k]["strict_edge_face_crossings"] > 0
                for row in rows
            ),
        }
    (root / "confirmation.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, allow_nan=False)
        + "\n"
    )
    print(summary, flush=True)


def timing(root):
    # Warm kernel execution, pin a single available logical CPU and include candidate application,
    # simulation, release, native frames, scoring and plan selection on both paths.
    torch.set_num_threads(1)
    core = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {core})
    net = FoldResponse(root / "fold-response.npz").eval()

    def learned(seed, landing, act):
        return get_choices(net, geometry(seed), act, landing, seed)

    seed, landing, act = task(0, 50300)
    learned(seed, landing, act)
    teacher(seed, act[1])
    measurements = []
    for i in range(4):
        seed, landing, act = task(i, 50300)
        g = geometry(seed)
        run = []
        for _ in range(10):
            begin = time.perf_counter()
            selected, _short, _, _, _ = learned(seed, landing, act)
            run.append(time.perf_counter() - begin)
        times = {}
        chosen = {}
        costs = {}
        arm_order = (
            ("full_newton", "hybrid") if i % 2 == 0 else ("hybrid", "full_newton")
        )
        for arm in arm_order:
            begin = time.perf_counter()
            if arm == "hybrid":
                selected, ids, _, _, _ = learned(seed, landing, act)
            else:
                ids = list(range(len(act)))
            objectives = []
            for j in ids:
                q, _ = teacher(seed, act[j])
                objectives.append(score(metrics(q, g, landing), g))
            chosen[arm] = int(ids[np.argmin(objectives)])
            times[arm] = time.perf_counter() - begin
            costs[arm] = dict(zip(ids, objectives))
        assert chosen["hybrid"] == min(
            costs["hybrid"], key=lambda j: costs["full_newton"][j]
        )
        measurements.append(
            {
                "seed": seed,
                "neural_seconds": float(np.median(run)),
                "full_newton_seconds": times["full_newton"],
                "hybrid_seconds": times["hybrid"],
                "full_choice": chosen["full_newton"],
                "hybrid_choice": chosen["hybrid"],
                "neural_choice": int(selected),
                "method_order": arm_order,
            }
        )
    clock = {
        "one_cpu_logical_core": core,
        "candidate_count": len(act),
        "hybrid_teacher_rollouts": 3,
        "timed_tasks": measurements,
    }
    for k in ("neural_seconds", "hybrid_seconds", "full_newton_seconds"):
        clock["mean_" + k] = float(np.mean([r[k] for r in measurements]))
    clock["full_over_hybrid"] = (
        clock["mean_full_newton_seconds"] / clock["mean_hybrid_seconds"]
    )
    clock["full_over_neural"] = (
        clock["mean_full_newton_seconds"] / clock["mean_neural_seconds"]
    )
    (root / "timings.json").write_text(json.dumps(clock, indent=2) + "\n")
    print(clock, flush=True)


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=["evaluate", "timing"])
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    a = cli.parse_args()
    (evaluate if a.mode == "evaluate" else timing)(a.root)
