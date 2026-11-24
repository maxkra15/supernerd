"""Export the Newton cable task candidate bank for offline action-submission scoring."""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from cable_gates import COMMANDS, GATES, SEEDS, Teacher


def export_geometry(job):
    seed, cpu, results = job
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    teacher = Teacher(seed)
    gate = np.stack([teacher.run(command)[GATES, -1, 1] for command in COMMANDS])
    # Verify every published candidate outcome against the directly measured bank.
    receipt = json.loads((results / f"object-{seed}.json").read_text())
    for goal in receipt["cases"]:
        target = np.asarray(goal["gate_y_m"], dtype=np.float32)
        for outcome in goal["methods"].values():
            measured = 1000 * np.abs(gate[outcome["candidate"]] - target)
            np.testing.assert_allclose(
                measured, outcome["actual_gate_distance_mm"], rtol=0.0, atol=0.001
            )
    return str(seed), {"commands": COMMANDS.tolist(), "gate_y_m": gate.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-offset", type=int, default=0)
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0))
    jobs = [
        (s, cpus[(args.cpu_offset + i % args.workers) % len(cpus)], args.results)
        for i, s in enumerate(SEEDS)
    ]
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        bank = dict(pool.map(export_geometry, jobs))
    args.output.write_text(json.dumps(bank, indent=2) + "\n")


if __name__ == "__main__":
    main()
