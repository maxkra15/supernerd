"""Generate symmetric Newton force perturbations for a local-sensitivity experiment."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pack_deformables import pack

DT = 0.02
FRAMES = 301
# Object geometry seeds are separate from the original shared model's fit, development and test seeds.
FOLDS = {"fit": range(26720, 26752), "probe": range(42760, 42768)}
AMPLITUDES = {"fit": (-0.5, 0.0, 0.5), "probe": (-0.5, -0.25, 0.0, 0.25, 0.5, 3.0)}


def force_direction(seed):
    rng = np.random.default_rng(np.random.SeedSequence([870241, seed]))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    onset, duration = rng.uniform(0.65, 1.85), rng.uniform(0.8, 1.8)
    phase = np.clip((np.arange(FRAMES) * DT - onset) / duration, 0.0, 1.0)
    cycles = rng.uniform(0.4, 1.7)
    signal = np.sin(np.pi * phase) ** 2 * np.cos(2 * np.pi * cycles * phase)
    return (signal[:, None] * axis / np.abs(signal).max()).astype(np.float32), onset


def capture(task):
    from teacher.teachers import generate_episode

    dest, fold, family, seed = task
    direction, onset = force_direction(seed)
    shape, native = [], []
    levels = AMPLITUDES[fold]
    source = {}
    for amplitude in levels:
        trajectory = generate_episode(
            family,
            seed,
            frames=FRAMES,
            dt=DT,
            device="cpu",
            additional_load=amplitude * direction,
        )
        record = dict(
            trajectory["metadata"],
            family=family,
            seed=seed,
            identity=f"{family}-{seed}",
            excitation="smooth directional probe",
            sha256="",
        )
        sample = pack(record, trajectory)
        if amplitude == 0:
            source = {
                k: sample[k] for k in ("static", "u", "family", "seed", "identity")
            }
            source.update(
                q0=sample["q"][0],
                v0=sample["v"][0],
                onset=onset,
                direction=direction[:-1],
                rest=trajectory["rest_position"],
                free=trajectory["inverse_mass"] > 0,
            )
        shape.append(sample["q"][1:])
        if fold == "probe":
            native.append(trajectory["position"][1:])
    source.update(amplitude=np.array(levels), q=np.stack(shape))
    if native:
        source["native"] = np.stack(native)
    np.savez_compressed(dest / fold / f"{family}-{seed}.npz", **source)
    return len(levels)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--fold", choices=("fit", "probe", "both"), default="both")
    args = cli.parse_args()
    start = time.perf_counter()
    tasks = []
    for fold, seeds in FOLDS.items():
        if args.fold != "both" and fold != args.fold:
            continue
        (args.output / fold).mkdir(parents=True, exist_ok=True)
        tasks.extend(
            (args.output, fold, family, seed)
            for seed in seeds
            for family in ("cloth", "cable")
        )
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        counts = list(pool.map(capture, tasks))
    report = {
        "objects": len(counts),
        "trajectories": sum(counts),
        "levels_m_s2": AMPLITUDES,
        "fit_objects_per_family": 32 if args.fold != "probe" else 0,
        "probe_objects_per_family": 8 if args.fold != "fit" else 0,
        "frames": FRAMES,
        "dt_s": DT,
        "workers": args.workers,
        "selected_fold": args.fold,
        "capture_including_compression_wall_s": time.perf_counter() - start,
    }
    (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
