"""Fit per-asset Gaussian radial basis networks to a nominal Newton trajectory.

Control-history centers are calibration trajectories at the same elapsed time. A
leave-one-program-out calibration score selects bandwidth and ridge strength. The
zero-control kernel is centered analytically to preserve the reference response.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from .asset_response import (
    DT, EVAL_COUNT, FRAMES, TRAIN_COUNT, TRAIN_SIZES, controls, features,
    metrics, summarize,
)


def kernel(z, centers, bandwidth):
    # Arrays have axes (phase, program, feature); every feature uses past or current control.
    delta = (z[:, :, None, :] - centers[:, None, :, :]) / bandwidth
    origin = np.exp(-0.5 * np.mean((z / bandwidth) ** 2, axis=-1))
    landmark_origin = np.exp(-0.5 * np.mean((centers / bandwidth) ** 2, axis=-1))
    return np.exp(-0.5 * np.mean(delta ** 2, axis=-1)) - origin[:, :, None] - landmark_origin[:, None, :] + 1


def fit(actions, displacement):
    centers = features(actions, "stationary").transpose(1, 0, 2)
    # Scale from training data, never from the test trajectories.
    scale = np.maximum(np.sqrt(np.mean(centers ** 2, axis=(0, 1))), 0.5)
    centers /= scale
    target = displacement.reshape(len(actions), FRAMES - 1, -1).transpose(1, 0, 2)
    best = None
    for bandwidth in (0.4, 0.85, 1.75, 4.0):
        gram = kernel(centers, centers, bandwidth)
        for ridge in (0.005, 0.03, 0.15):
            weights = gram + np.eye(len(actions))[None, :, :] * ridge
            # Exact leave-one-program-out prediction errors for kernel ridge regression.
            inverse = np.linalg.inv(weights)
            alpha = inverse @ target
            diagonal = np.diagonal(inverse, axis1=1, axis2=2)
            residual = alpha / diagonal[:, :, None]
            validation = float(np.mean(residual ** 2))
            if best is None or validation < best[0]:
                best = (validation, bandwidth, ridge, alpha)
    validation, bandwidth, ridge, alpha = best
    return {"centers": centers, "scale": scale, "bandwidth": bandwidth, "ridge": ridge,
            "alpha": alpha, "leave_one_out_mse": validation}


def predict(model, actions, nominal):
    z = features(actions, "stationary").transpose(1, 0, 2) / model["scale"]
    weights = kernel(z, model["centers"], model["bandwidth"])
    pred = (weights @ model["alpha"]).transpose(1, 0, 2)
    return nominal + pred.reshape(len(actions), *nominal.shape[1:])


def fresh(record):
    from ..teacher.teachers import generate_episode

    family, seed, output = record
    start = perf_counter()
    repeated = generate_episode(family, seed, frames=FRAMES, dt=DT)["position"]
    repeated_u = controls(seed, 0, True)
    repeated0 = generate_episode(family, seed, frames=FRAMES, dt=DT, additional_load=repeated_u)["position"][1:]
    xs, ys = [], []
    for idx in range(EVAL_COUNT):
        u = controls(seed, EVAL_COUNT + idx, True)
        obs = generate_episode(family, seed, frames=FRAMES, dt=DT, additional_load=u)
        xs.append(u[:-1])
        ys.append(obs["position"][1:])
        np.testing.assert_array_equal(repeated[0], obs["position"][0])
    np.savez_compressed(output / f"fresh-{family}-{seed}.npz", repeat=repeated,
                        repeat0=repeated0, u=xs, y=ys)
    return perf_counter() - start


def evaluate(record, base, output):
    family, seed = record["family"], record["seed"]
    with np.load(base/f"{family}-{seed}.npz") as train, np.load(output/f"fresh-{family}-{seed}.npz") as test:
        nominal = train["nominal"][None, 1:].astype(np.float64)
        y = test["y"].astype(np.float64)
        u = test["u"]
        errors = {}
        for n in TRAIN_SIZES:
            start = perf_counter()
            model = fit(train["train_u"][:n], train["train_q"][:n].astype(np.float64) - nominal)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            prediction = predict(model, u, nominal)
            predict_seconds = perf_counter() - start
            rows = metrics(prediction, y, nominal, record["asset_span_m"])
            errors[str(n)] = {"fit_s": fit_seconds, "batch_prediction_s": predict_seconds, "test": rows,
                              "mean": summarize(rows), "bandwidth": model["bandwidth"], "ridge": model["ridge"],
                              "loo_mse": model["leave_one_out_mse"]}
            if n == TRAIN_COUNT:
                np.savez_compressed(output/f"weights-{family}-{seed}.npz", nominal=nominal[0], **model)
                np.savez_compressed(output/f"visual-{family}-{seed}.npz", nominal=nominal[0], target=y[0],
                                    predicted=prediction[0], force=u[0])
        q_err = lambda a: float(np.sqrt(np.mean(np.sum(a.astype(np.float64) ** 2, axis=-1))) * 1e3)
        return {"family": family, "seed": seed, "nodes": record["nodes"], "asset_span_m": record["asset_span_m"],
                "capture_s": record["capture_seconds"], "nominal": summarize(metrics(nominal, y, nominal,
                                                                                      record["asset_span_m"])),
                "nominal_repeat_rms_mm": q_err(test["repeat"] - train["nominal"]),
                "old_test0_repeat_rms_mm": q_err(test["repeat0"] - train["test_q"][0]), "fit": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.base/"results.json").read_text())
    records = data["assets"]
    work = [(r["family"], r["seed"], args.output) for r in records]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fresh, work))
    fresh_wall = perf_counter()-start
    models = [evaluate(r, args.base, args.output) for r in records]
    result = {}
    for family in ("cable", "cloth"):
        subset = [r for r in models if r["family"]==family]
        curve = {}
        for n in TRAIN_SIZES:
            rows = [item for r in subset for item in r["fit"][str(n)]["test"]]
            curve[str(n)] = {**summarize(rows),
                             "capture_s_median_per_asset": float(np.median([r["capture_s"]["nominal"] +
                                                                           sum(r["capture_s"]["train"][:n]) for r in subset])),
                             "fit_ms_median_per_asset": float(np.median([r["fit"][str(n)]["fit_s"] for r in subset]) * 1e3),
                             "better_than_nominal": sum(t["percent_response"] < 100 for t in rows),
                             "programs": len(rows)}
        result[family] = {"fit": curve, "nominal": summarize([r["nominal"] for r in subset]),
                          "max_repeat_rms_mm": max(max(r["nominal_repeat_rms_mm"],r["old_test0_repeat_rms_mm"]) for r in subset)}
    receipt = {"method": "Phase-specific causal Gaussian RBF network with train-only leave-one-program-out bandwidth/ridge selection",
               "test": "new seed bank of 8 programs on each calibrated asset", "fresh_capture_wall_s": fresh_wall,
               "results": result, "assets": models}
    (args.output/"results.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
