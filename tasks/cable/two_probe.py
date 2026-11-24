"""Test causal two-probe identification with and without a learned memory coefficient."""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cable_gates import COMMANDS, GATES, SEEDS, TOLERANCE, ModalForward, Teacher, pack
from fast_planner import fast_modal
from scipy.optimize import lsq_linear


def response_matrix(diagonal, memory_share):
    # The second pulse starts after gate one. Its first-gate derivative is exactly zero.
    return np.array(
        [
            [diagonal[0], 0.0],
            [diagonal[1] * memory_share, diagonal[1] * (1.0 - memory_share)],
        ]
    )


def geometry(args):
    seed, directory, weights, cpu = args
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    obj = Teacher(seed)
    ep = obj.episode
    context = pack(
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
    model = ModalForward().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        fast_modal(model, context, ep, obj.shapes)
        clocks = []
        for _ in range(5):
            start = perf_counter()
            prediction = fast_modal(model, context, ep, obj.shapes)
            first = prediction[17, 1] - prediction[12, 1]
            second = prediction[13, 1] - prediction[12, 1]
            share = (
                float(first / (first + second)) if abs(first + second) > 1e-8 else 0.0
            )
            clocks.append(perf_counter() - start)
    assert np.array_equal(COMMANDS[[12, 17, 13]], [[0, 0], [5, 0], [0, 5]])
    start = perf_counter()
    nominal = obj.run([0, 0])[GATES, -1, 1].astype(float)
    joint = obj.run([5, 5])[GATES, -1, 1].astype(float)
    diagonal = (joint - nominal) / 5
    teacher_time = perf_counter() - start
    grids = json.loads((directory / f"object-{seed}.json").read_text())["cases"]
    rows = []
    for item in grids:
        target = np.array(item["gate_y_m"])
        record = {}
        for method, ratio in (("newton_no_memory", 0.0), ("supernerd_memory", share)):
            start = perf_counter()
            amp = lsq_linear(
                response_matrix(diagonal, ratio), target - nominal, bounds=(-10.0, 10.0)
            ).x
            plan = (
                teacher_time
                + (np.median(clocks) if method == "supernerd_memory" else 0.0)
                + (perf_counter() - start)
            )
            start = perf_counter()
            actual = obj.run(amp)[GATES, -1, 1]
            total = plan + perf_counter() - start
            err = float(np.abs(actual - target).max())
            record[method] = {
                "amplitudes": amp.tolist(),
                "peak_error_mm": err * 1000,
                "success": err <= TOLERANCE,
                "plan_s": float(plan),
                "total_s": float(total),
            }
        rows.append(record)
    return {
        "seed": seed,
        "nominal_y_m": nominal.tolist(),
        "diagonal_increment_m_per_ms2": diagonal.tolist(),
        "supernerd_memory_share": share,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-offset", type=int, default=0)
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0))
    work = [
        (
            s,
            args.results,
            args.mixed,
            cpus[(args.cpu_offset + i % args.workers) % len(cpus)],
        )
        for i, s in enumerate(SEEDS)
    ]
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        cases = list(pool.map(geometry, work))
    all_cases = [row for g in cases for row in g["cases"]]
    summary = {}
    for method in all_cases[0]:
        rows = [r[method] for r in all_cases]
        summary[method] = {
            "passed": sum(r["success"] for r in rows),
            "mean_worst_error_mm": float(np.mean([r["peak_error_mm"] for r in rows])),
            "median_plan_s": float(np.median([r["plan_s"] for r in rows])),
            "median_total_s": float(np.median([r["total_s"] for r in rows])),
        }
    args.output.write_text(
        json.dumps({"summary": summary, "geometries": cases}, indent=2) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
