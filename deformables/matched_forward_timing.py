"""Time batch-one CPU rollouts with native XYZ output on fresh Newton geometries."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from modal_forward import ModalForward, load
from pack_deformables import interpolation
from teacher import teachers
from teacher.elastic_load_diversity import _held_load


def make_recorder(rest, family, expected, measurement):
    def run(model, solver, dt, values, force_array, decode, *, force_field, substeps):
        first, second = model.state(), model.state()
        control = model.control()
        local = np.zeros((len(rest) - 1, 3), dtype=np.float32)
        local[:, 2] = 0.5 * np.linalg.norm(np.diff(rest, axis=0), axis=1)
        output = np.empty_like(expected)

        def observe(state, xyz):
            if family == "cloth":
                xyz[:] = state.particle_q.numpy()
            else:
                pose = state.body_q.numpy()
                center, axis, scalar = pose[:, :3], pose[:, 3:6], pose[:, 6:]
                offset = local + 2 * np.cross(
                    axis, np.cross(axis, local) + scalar * local
                )
                xyz[0] = center[0] - offset[0]
                xyz[1:] = center + offset

        beginning = time.perf_counter()
        for i, value in enumerate(values[:-1]):
            force_array.assign(value)
            first, second, _ = teachers._advance(
                first,
                second,
                control,
                solver,
                dt,
                force_array,
                force_field,
                substeps,
                None,
                None,
            )
            observe(first, output[i])
        measurement["newton_s"] = time.perf_counter() - beginning
        measurement["teacher_replay_max_m"] = float(np.max(np.abs(output - expected)))
        if not np.allclose(output, expected, atol=5e-6, rtol=0):
            raise ValueError("Newton trajectory differs from archived native positions")
        raise Finished

    return run


class Finished(Exception):
    """Stop the teacher after the benchmark recorder returns its timing."""


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/matched-forward-timing.json")
    )
    parser.add_argument("--core", type=int, default=min(os.sched_getaffinity(0)))
    args = parser.parse_args()
    os.sched_setaffinity(0, [args.core])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = ModalForward().eval()
    model.load_state_dict(
        torch.load("weights/modal-paired.pt", weights_only=True, map_location="cpu")
    )
    samples = load(Path("data/fresh.npz"), "cpu")
    native = np.load("data/native-fresh.npz")
    clock = np.arange(301) * 0.02
    extra, _ = _held_load(clock, "descending-chirp", 6)
    cases = []
    ids = [
        i
        for i, name in enumerate(samples["identity"])
        if name.endswith("descending-chirp-6")
    ]
    with wp.ScopedDevice("cpu"):
        for i in [ids[0], ids[-1]] + ids:
            name = str(samples["identity"][i])
            seed = int(samples["seed"][i])
            family = str(samples["family"][i])
            result = {"identity": name}
            rest = native[name + "_rest"]
            free = native[name + "_free"]
            target = native[name][1:]
            scene = teachers._cloth if family == "cloth" else teachers._cable
            try:
                scene(
                    seed,
                    np.random.default_rng(seed),
                    clock,
                    0.02,
                    "cpu",
                    additional_load=extra,
                    recorder=make_recorder(rest, family, target, result),
                )
            except Finished:
                pass
            static = samples["static"][i : i + 1]
            axial = interpolation(5, int(static[0, 3]) + 1)
            lateral = (
                interpolation(3, int(static[0, 4]) + 1)
                if family == "cloth"
                else np.full((1, 3), 1 / 3)
            )
            projector = (float(static[0, 1]) * np.kron(lateral, axial)).astype(
                np.float32
            )
            projector[~free] = 0
            fixed, carry = model.start(
                samples["q"][i : i + 1, 0], samples["v"][i : i + 1, 0], static
            )
            controls = samples["u"][i].numpy()
            output = np.empty_like(target)
            beginning = time.perf_counter()
            for j, u in enumerate(controls):
                carry, predicted = model.advance(
                    fixed, carry, torch.from_numpy(u[None])
                )
                output[j] = rest + projector @ predicted[0].numpy()
            result["neural_s"] = time.perf_counter() - beginning
            result["neural_rms_m"] = float(
                np.sqrt(np.mean(np.sum((output - target) ** 2, axis=-1)))
            )
            cases.append(result)
    cases = cases[2:]
    by_family = {}
    for family in ("cloth", "cable"):
        rows = [r for r in cases if r["identity"].startswith(family)]
        teacher = sum(r["newton_s"] for r in rows)
        neural = sum(r["neural_s"] for r in rows)
        intervals = 300 * len(rows)
        by_family[family] = {
            "newton_ms": teacher * 1000 / intervals,
            "neural_ms": neural * 1000 / intervals,
            "speed_ratio": teacher / neural,
        }
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "cpu_core": args.core,
                "batch": 1,
                "dt": 0.02,
                "steps_per_object": 300,
                "profile": "descending chirp, 6 m/s^2",
                "method": "CPU affinity one core. One untimed warmup per family. Precomputed control schedules and native geometry maps. Force assignment, full integration, native XYZ decode, and writing every frame to host memory timed for both. Scene creation, files, model initialization excluded.",
                "case_measurements": cases,
                "summary": by_family,
            },
            indent=2,
        )
        + "\n"
    )
    print(by_family)


if __name__ == "__main__":
    main()
