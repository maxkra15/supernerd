"""Identify cable and cloth dynamics about a fixed asset release from causal force history."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np



FRAMES = 81
DT = 0.02
TRAIN_COUNT = 12
EVAL_COUNT = 8
TRAIN_SIZES = (2, 4, 8, 12)
LAG_KNOTS = np.array([0, 1, 3, 6, 10, 16, 24, 36, 52, 79])
PHASE_KNOTS = np.array([0, 0.22, 0.50, 0.78, 1])
METHODS = ("stationary", "phase", "neural")
RIDGE = 0.002
TEST_NAMES = ("reversal", "chirp", "late_pulse", "rotating")


def controls(seed, index, test):
    rng = np.random.default_rng([seed, 2 if test else 1, index])
    t = np.arange(FRAMES) * DT
    directions = rng.normal(size=(4, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitude = rng.uniform(3.5, 6.0)
    if not test:
        # Training loads vary in direction, phase, frequency and support.
        knots = np.linspace(0, t[-1], 7)
        node = rng.uniform(-1, 1, (len(knots), 3))
        node[[0, -1]] = 0
        indices = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
        fraction = (t - knots[indices]) / np.diff(knots)[indices]
        blend = fraction ** 2 * (3 - 2 * fraction)
        u = node[indices] * (1 - blend[:, None]) + node[indices + 1] * blend[:, None]
        u += 0.35 * np.sin(2 * np.pi * rng.uniform(0.5, 2.8) * t)[:, None] * directions[0]
    else:
        kind = TEST_NAMES[index % len(TEST_NAMES)]
        envelope = np.sin(np.pi * t / t[-1]) ** 2
        if kind == "reversal":
            u = envelope[:, None] * np.tanh((0.8 - t) / 0.08)[:, None] * directions[0]
        elif kind == "chirp":
            u = envelope[:, None] * np.sin(2 * np.pi * (0.25 * t + 0.95 * t ** 2))[:, None] * directions[0]
        elif kind == "late_pulse":
            phase = np.clip((t - 0.8) / 0.38, 0, 1)
            u = np.sin(np.pi * phase)[:, None] ** 2 * directions[1]
        else:
            angle = 2 * np.pi * 0.8 * t
            u = envelope[:, None] * (np.sin(angle)[:, None] * directions[1] +
                                      np.cos(angle)[:, None] * directions[2])
    return (magnitude * u).astype(np.float32)


def capture(task):
    from ..teacher.teachers import generate_episode

    family, seed, out = task
    start = perf_counter()
    base = generate_episode(family, seed, frames=FRAMES, dt=DT, device="cpu")
    nominal = base["position"]
    arrays = {"nominal": nominal, "free_nodes": (base["inverse_mass"] > 0)}
    costs = {"nominal": perf_counter() - start}
    for test, prefix, count in ((False, "train", TRAIN_COUNT), (True, "test", EVAL_COUNT)):
        actions, positions, elapsed = [], [], []
        for i in range(count):
            start = perf_counter()
            u = controls(seed, i, test)
            rec = generate_episode(family, seed, frames=FRAMES, dt=DT, device="cpu", additional_load=u)
            np.testing.assert_array_equal(rec["position"][0], nominal[0])
            positions.append(rec["position"][1:])
            actions.append(u[:-1])
            elapsed.append(perf_counter() - start)
        arrays[prefix + "_u"] = np.stack(actions)
        arrays[prefix + "_q"] = np.stack(positions)
        costs[prefix] = elapsed
    if int(base["metadata"]["seed"]) != seed:
        raise RuntimeError("Unexpected randomized asset")
    np.savez_compressed(out / f"{family}-{seed}.npz", **arrays)
    return {"family": family, "seed": seed, "nodes": len(nominal[0]), "metadata": base["metadata"],
            "capture_seconds": costs}


def hat(x, knots):
    identity = np.eye(len(knots))
    return np.stack([np.interp(x, knots, values) for values in identity], axis=-1)


def features(actions, method):
    # tap 0 encodes u[t], which drives the prediction of q[t+1]
    impulse = hat(np.arange(FRAMES - 1), LAG_KNOTS)
    impulse /= impulse.sum(axis=0)
    history = np.zeros((len(actions), FRAMES - 1, 3, len(LAG_KNOTS)))
    for lag in range(FRAMES - 1):
        history[:, lag:] += actions[:, : FRAMES - 1 - lag, :, None] * impulse[lag]
    x = history.reshape(len(actions), FRAMES - 1, -1)
    if method == "stationary":
        return x
    phase = hat(np.linspace(0, 1, FRAMES - 1), PHASE_KNOTS)
    varying = (x[..., None] * phase[None, :, None, :]).reshape(len(actions), FRAMES - 1, -1)
    if method == "phase":
        return varying
    # Frozen random tanh features provide a one-hidden-layer neural baseline. Subtract
    # the zero-control activation so every model reproduces the known reference exactly.
    rng = np.random.default_rng(100)
    matrix = rng.normal(size=(x.shape[-1], 48)) / np.sqrt(x.shape[-1])
    bias = rng.normal(scale=0.5, size=(len(PHASE_KNOTS), 48))
    offset = phase @ bias
    nonlinear = np.tanh(x @ matrix + offset) - np.tanh(offset)
    return np.concatenate((varying, nonlinear), axis=-1)


def fit(actions, response, method):
    design = features(actions, method)
    design = design.reshape(-1, design.shape[-1])
    target = response.reshape(len(design), -1).astype(np.float64)
    scale = np.maximum(np.sqrt(np.mean(design ** 2, axis=0)), 0.02)
    design /= scale
    gram = design.T @ design
    gram.flat[::len(gram)+1] += len(design) * RIDGE
    coeff = np.linalg.solve(gram, design.T @ target)
    return coeff / scale[:, None]


def metrics(predicted, target, nominal, span):
    rms = lambda q: np.sqrt(np.mean(np.sum(q ** 2, axis=-1), axis=(1, 2)))
    error = rms(predicted - target)
    response = rms(target - nominal)
    rows = []
    for i in range(len(target)):
        rows.append({"program": TEST_NAMES[i % len(TEST_NAMES)], "test_index": i,
                     "rms_mm": float(error[i] * 1e3), "percent_span": float(100 * error[i] / span),
                     "percent_response": float(100 * error[i] / max(response[i], 1e-12)),
                     "target_response_mm": float(response[i] * 1e3)})
    return rows


def summarize(rows):
    return {k: float(np.mean([v[k] for v in rows])) for k in
            ("rms_mm", "percent_span", "percent_response", "target_response_mm")}


def assess(record, output):
    path = output / f"{record['family']}-{record['seed']}.npz"
    with np.load(path) as data:
        nominal = data["nominal"][None, 1:].astype(np.float64)
        span = float(np.linalg.norm(np.ptp(data["nominal"][0], axis=0)))
        train_u, train_q = data["train_u"], data["train_q"]
        test_u, test_q = data["test_u"], data["test_q"].astype(np.float64)
        assert not any(np.array_equal(a,b) for a in train_u for b in test_u)
        curves = {}
        coeffs = {"nominal": data["nominal"]}
        visual = {"nominal": nominal[0], "test": test_q[0]}
        for method in METHODS:
            method_curve = {}
            for n in TRAIN_SIZES:
                start = perf_counter()
                coeff = fit(train_u[:n], train_q[:n].astype(np.float64) - nominal, method)
                fit_ms = (perf_counter() - start) * 1000
                start = perf_counter()
                response = features(test_u, method) @ coeff
                pred = nominal + response.reshape(test_q.shape)
                prediction_ms = (perf_counter() - start) * 1000
                rows = metrics(pred, test_q, nominal, span)
                method_curve[str(n)] = {"fit_ms": fit_ms, "prediction_ms_for_all_tests": prediction_ms,
                                        "features": coeff.shape[0], "tests": rows, "mean": summarize(rows)}
                if n == TRAIN_COUNT:
                    coeffs[method] = coeff
                    visual[method] = pred[0]
            curves[method] = method_curve
        np.savez_compressed(output / f"weights-{record['family']}-{record['seed']}.npz", **coeffs)
        np.savez_compressed(output / f"visual-{record['family']}-{record['seed']}.npz", **visual)
        record.update({"asset_span_m": span, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "nominal_baseline": summarize(metrics(nominal, test_q, nominal, span)),
                       "fit_curve": curves})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--assets-per-family", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(modality, 20483100 + idx + j*100, args.output)
             for j, modality in enumerate(("cable", "cloth")) for idx in range(args.assets_per_family)]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(capture, tasks))
    elapsed = perf_counter() - start
    scored = [assess(rec, args.output) for rec in records]
    summary = {}
    for family in ("cable", "cloth"):
        subset = [v for v in scored if v["family"] == family]
        settings = {}
        for method in METHODS:
            sizes = {}
            for n in TRAIN_SIZES:
                rows = [v["fit_curve"][method][str(n)] for v in subset]
                test_rows = [item for row in rows for item in row["tests"]]
                source_cost = [v["capture_seconds"]["nominal"] + sum(v["capture_seconds"]["train"][:n])
                               for v in subset]
                sizes[str(n)] = {**summarize(test_rows),
                                 "fit_ms_median": float(np.median([r["fit_ms"] for r in rows])),
                                 "capture_calibration_s_per_asset_median": float(np.median(source_cost)),
                                 "fraction_better_than_nominal": float(np.mean([t["percent_response"] < 100 for t in test_rows])),
                                 "worst_case_percent_response": float(max(t["percent_response"] for t in test_rows))}
            settings[method] = sizes
        summary[family] = {"assets": len(subset), "test_count": len(subset)*EVAL_COUNT,
                           "baseline": summarize([v["nominal_baseline"] for v in subset]), "models": settings}
    result = {"task": "Contact-free, fixed-release per-asset forced forward response about stored nominal trajectory.",
              "frames_predicted": FRAMES - 1, "dt": DT, "causal_control_known": True,
              "nominal_training_trajectory_at_inference": True, "workers": args.workers,
              "capture_wall_s_including_all_training_test": elapsed,
              "additional_fit_programs": list(TRAIN_SIZES), "test_program_families": list(TEST_NAMES),
              "ridge": RIDGE, "results": summary, "assets": scored}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
