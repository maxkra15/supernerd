"""Continuous two-gate control and numerical-teacher sensitivity on a frozen grid of inputs."""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cable_gates import (
    COMMANDS,
    GATES,
    SEEDS,
    TOLERANCE,
    ModalForward,
    PortableRecurrent,
    Teacher,
    modal_predict,
    neurall_predict,
    pack,
    targets,
)
from scipy.optimize import lsq_linear

TRIANGLES = np.array(
    [
        [i, j, k]
        for r in range(4)
        for c in range(4)
        for i, j, k in (
            (5 * r + c, 5 * r + c + 1, 5 * r + c + 6),
            (5 * r + c, 5 * r + c + 5, 5 * r + c + 6),
        )
    ]
)


def inverse(table, target):
    """Closest reachable point in the piecewise-linear approximation of the two-dimensional gate map."""
    estimates = []
    for ids in TRIANGLES:
        q = table[ids]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            d = q[j] - q[i]
            alpha = np.clip(np.dot(target - q[i], d) / max(np.dot(d, d), 1e-20), 0, 1)
            estimates.append(
                (
                    q[i] + alpha * d,
                    COMMANDS[ids[i]] + alpha * (COMMANDS[ids[j]] - COMMANDS[ids[i]]),
                )
            )
        mat = np.stack((q[1] - q[0], q[2] - q[0]), axis=1)
        if abs(np.linalg.det(mat)) < 1e-12:
            continue
        w = np.linalg.solve(mat, target - q[0])
        if np.all(w >= 0) and w.sum() <= 1:
            estimates.append((target, np.r_[1 - w.sum(), w] @ COMMANDS[ids]))
    return min(
        estimates, key=lambda x: (np.linalg.norm(x[0] - target), np.linalg.norm(x[1]))
    )[1]


def compact_prediction(model, sample, ep, teacher, mode, subs):
    loads = np.broadcast_to(sample["u"], (25, *sample["u"].shape)).copy()
    loads[:, :, 1] += COMMANDS @ teacher.shapes[:-1].T
    if mode == "neurall":
        pos = neurall_predict(subs, model, sample, ep, loads)
    else:
        pos = modal_predict(model, sample, ep, loads)
    return pos[:, GATES - 1, -1, 1].astype(float).copy()


def solve_case(job):
    seed, options, cpu = job
    out, parent, mixed, nrpath = map(Path, options)
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    obj = Teacher(seed)
    ep = obj.episode
    sample = pack(
        dict(
            ep["metadata"],
            family="cable",
            seed=seed,
            identity="held",
            excitation="gates",
            sha256="",
        ),
        ep,
    )
    tables = {}
    clocks = {}
    with torch.inference_mode():
        for name, p in (
            ("supernerd", parent),
            ("supernerd_mixed", mixed),
            ("neurall", nrpath),
        ):
            bundle = torch.load(p, weights_only=True, map_location="cpu")
            if name == "neurall":
                model = PortableRecurrent(bundle).eval()
            else:
                model = ModalForward().eval()
                model.load_state_dict(bundle)
            compact_prediction(model, sample, ep, obj, name, bundle)
            times = []
            for _ in range(5):
                start = perf_counter()
                tables[name] = compact_prediction(model, sample, ep, obj, name, bundle)
                times.append(perf_counter() - start)
            clocks[name] = float(np.median(times))
    # The direct shooting table is also used as a reservoir for three calibration rollouts,
    # with the actual measured constituent times charged separately to the three-query planner.
    clock = []
    stock = []
    for cmd in COMMANDS:
        start = perf_counter()
        stock.append(obj.run(cmd)[GATES, -1, 1].astype(float))
        clock.append(perf_counter() - start)
    tables["newton"] = np.stack(stock)
    clocks["newton"] = sum(clock)
    z = 12
    first = 17  # [+5,0]
    second = 13  # [0,+5]
    assert np.all(COMMANDS[[z, first, second]] == [[0, 0], [5, 0], [0, 5]])
    nom = tables["newton"][z]
    J = (tables["newton"][[first, second]] - nom).T / 5
    rows = []
    for j, witness in enumerate(targets(seed)):
        secret = obj.run(witness)
        target = secret[GATES, -1, 1].astype(float)
        new_ref = obj.run(witness, refined=True)[GATES, -1, 1].astype(float)
        row = {
            "target_gate_m": target.tolist(),
            "refined_witness_gate_m": new_ref.tolist(),
            "methods": {},
        }
        for name in (
            "supernerd",
            "supernerd_mixed",
            "neurall",
            "newton",
            "supernerd_one_replay",
            "newton_three_probes",
        ):
            start = perf_counter()
            if name == "supernerd_one_replay":
                table = tables["supernerd_mixed"] - tables["supernerd_mixed"][z] + nom
                cmd = inverse(table, target)
                bank = clocks["supernerd_mixed"] + clock[z]
            elif name == "newton_three_probes":
                cmd = lsq_linear(J, target - nom, bounds=(-10, 10)).x
                bank = sum(clock[i] for i in (z, first, second))
            else:
                cmd = inverse(tables[name], target)
                bank = clocks[name]
            plan = bank + perf_counter() - start
            started = perf_counter()
            pose = obj.run(cmd)
            replay = perf_counter() - started
            gate = pose[GATES, -1, 1]
            refined = obj.run(cmd, refined=True)[GATES, -1, 1]
            peak = float(np.abs(gate - target).max())
            refined_peak = float(np.abs(refined - target).max())
            refined_paired = float(np.abs(refined - new_ref).max())
            row["methods"][name] = {
                "candidate": cmd.tolist(),
                "newton_peak_mm": 1000 * peak,
                "pass": peak <= TOLERANCE,
                "refined_stock_goal_peak_mm": 1000 * refined_peak,
                "refined_stock_goal_pass": refined_peak <= TOLERANCE,
                "refined_witness_goal_peak_mm": 1000 * refined_paired,
                "refined_witness_goal_pass": refined_paired <= TOLERANCE,
                "planner_s": plan,
                "planner_and_replay_s": plan + replay,
            }
        rows.append(row)
    (out / f"continuous-{seed}.json").write_text(json.dumps(rows, indent=2) + "\n")
    return seed


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", required=True, type=Path)
    cli.add_argument("--mixed", required=True, type=Path)
    cli.add_argument("--neurall", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--cpu-offset", type=int, default=0)
    args = cli.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    options = (args.output, args.parent, args.mixed, args.neurall)
    cpus = sorted(os.sched_getaffinity(0))
    work = [
        (seed, options, cpus[(args.cpu_offset + i % args.workers) % len(cpus)])
        for i, seed in enumerate(SEEDS)
        if not (args.output / f"continuous-{seed}.json").exists()
    ]
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for seed in pool.map(solve_case, work):
            print(seed, flush=True)
    rows = [
        r
        for s in SEEDS
        for r in json.loads((args.output / f"continuous-{s}.json").read_text())
    ]
    diff = [
        1000
        * np.abs(np.subtract(r["target_gate_m"], r["refined_witness_gate_m"])).max()
        for r in rows
    ]
    summary = {
        "newton_refined_witness_still_meets_stock_target": int(
            np.sum(np.asarray(diff) <= 10)
        ),
        "stock_refined_witness_mean_worst_gate_shift_mm": float(np.mean(diff)),
        "methods": {},
    }
    for name in rows[0]["methods"]:
        rr = [r["methods"][name] for r in rows]
        summary["methods"][name] = {
            **{
                k: int(sum(r[k] for r in rr))
                for k in (
                    "pass",
                    "refined_stock_goal_pass",
                    "refined_witness_goal_pass",
                )
            },
            **{
                k: float(np.median([r[k] for r in rr]))
                for k in ("planner_s", "planner_and_replay_s")
            },
            "mean_newton_worst_gate_mm": float(
                np.mean([r["newton_peak_mm"] for r in rr])
            ),
        }
    (args.output / "continuous-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
