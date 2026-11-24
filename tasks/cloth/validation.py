"""Check temporal and spatial sensitivity of representative physical folding outcomes."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import folding
import numpy as np
from experiment import task


def run(args):
    case, idx, mode, bank = args
    if mode == "finer":
        folding.NX = 24
        folding.NY = 16
    else:
        folding.NX = 12
        folding.NY = 8
    seed, landing, actions = task(case)
    q, _ = folding.teacher(seed, actions[idx], substeps=6, iterations=12)
    m = folding.metrics(q, folding.geometry(seed), landing)
    with np.load(bank / f"test-{case:02}-{idx:02}.npz") as orig:
        old = orig["q"]
        coarsened = (
            q.reshape(len(q), 17, 25, 3)[:, ::2, ::2].reshape(old.shape)
            if mode == "finer"
            else q
        )
        delta = coarsened[-1] - old[-1]
        change = float(np.sqrt((delta**2).sum(-1).mean()) * 1000)
    return {
        "case": case,
        "action_index": idx,
        "qualification": mode,
        "final_disagreement_mm": change,
        "metrics": m,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    init = json.loads((root / "results.json").read_text())
    tasks = [
        (i, init["cases"][i]["indices"]["model"], mode, root / "test")
        for i in range(4)
        for mode in ["time", "finer"]
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, tasks))
    (root / "teacher-refinement.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(rows)
