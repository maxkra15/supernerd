"""Plan and grade two lateral cable-tip gates using the same 25 actions for all models."""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from modal_forward import ModalForward
from native_rollouts import decode
from pack_deformables import pack
from scipy.interpolate import CubicSpline
from teacher import teachers
from torch import nn

DT = 0.02
FRAMES = 121
GATES = np.array((55, 120))
COMMANDS = np.array(list(product((-10.0, -5.0, 0.0, 5.0, 10.0), repeat=2)), np.float32)
SEEDS = tuple(range(51240, 51248))
TOLERANCE = 0.010


def pulses():
    time = np.arange(FRAMES) * DT
    result = np.zeros((FRAMES, 2), np.float32)
    for col, (start, width) in enumerate(((0.32, 0.72), (1.60, 0.70))):
        phase = np.clip((time - start) / width, 0, 1)
        result[:, col] = np.sin(np.pi * phase) ** 2
        result[:, col] /= result[:, col].max()
    return result


def targets(seed):
    rng = np.random.default_rng(842011 + seed)
    return np.array(list(product((-1, 1), repeat=2))) * rng.uniform(5, 9, (4, 2))


class Teacher:
    """Warm one Newton scene; reset solver and both physical state buffers per action."""

    def __init__(self, seed):
        ts = np.arange(FRAMES) * DT

        def capture(model, solver, dt, values, force_array, decode_native, **kw):
            self.model = model
            self.nominal_wrench = values.copy()
            self.mass = float(model.body_mass.numpy()[-1])
            self.force_field = kw["force_field"]
            self.force = force_array
            self.substeps = kw["substeps"]
            self.decode_native = decode_native
            return teachers._record(
                model, solver, dt, values, force_array, decode_native, **kw
            )

        with teachers.wp.ScopedDevice("cpu"):
            self.episode = teachers._cable(
                seed, np.random.default_rng(seed), ts, DT, "cpu", recorder=capture
            )
        self.shapes = pulses()

    def run(self, command, refined=False):
        model = self.model
        steps = 8 if refined else self.substeps
        solver = teachers.newton.solvers.SolverVBD(
            model, iterations=16 if refined else 8
        )
        a, b, control = model.state(), model.state(), model.control()
        wrench = self.nominal_wrench.copy()
        wrench[:, -1, 1] += self.mass * (
            self.shapes @ np.asarray(command, dtype=np.float32)
        )
        trajectory = np.empty_like(self.episode["position"])
        trajectory[0] = self.episode["position"][0]
        for i, value in enumerate(wrench[:-1]):
            self.force.assign(value)
            a, b, _ = teachers._advance(
                a,
                b,
                control,
                solver,
                DT,
                self.force,
                self.force_field,
                steps,
                None,
                None,
            )
            trajectory[i + 1] = self.decode_native(a)["position"]
        return trajectory


class PortableRecurrent(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        self.core = nn.GRU(len(bundle["center"]) + 3, bundle["width"], batch_first=True)
        self.shape = nn.Linear(bundle["width"], len(bundle["basis"]))
        self.load_state_dict(bundle["model"])


def neurall_predict(bundle, model, s, ep, loads):
    static = (torch.from_numpy(s["static"]) - bundle["center"]) / bundle["scale"]
    load = torch.from_numpy(loads)
    x = torch.cat((static.expand(*load.shape[:2], -1), load / 10), -1)
    latent = model.shape(model.core(x)[0])
    rest = torch.from_numpy(ep["rest_position"])
    size = len(rest) - 1
    decoder = CubicSpline(np.linspace(0, 1, 17), np.eye(17), axis=0)(
        np.linspace(0, 1, size)
    ).astype("f")
    decode_native = np.pad(
        np.kron(decoder, np.eye(3, dtype=np.float32)), ((3, 0), (0, 0))
    )
    decode_native[np.repeat(ep["inverse_mass"] == 0, 3)] = 0
    readout = torch.from_numpy(decode_native) @ bundle["basis"].T * s["length"]
    pos = rest + (latent @ readout.T).unflatten(-1, (len(rest), 3))
    if bundle["project_cable_length"]:
        delta = pos.diff(dim=2)
        ref = rest.diff(dim=0)
        norm = delta.norm(dim=-1, keepdim=True)
        projected = ref.norm(dim=-1, keepdim=True) * torch.where(
            norm > 1e-7,
            delta / norm.clamp_min(1e-7),
            ref / ref.norm(dim=-1, keepdim=True),
        )
        pos = torch.cat(
            (pos[:, :, :2], pos[:, :, 1:2] + projected[:, :, 1:].cumsum(2)), dim=2
        )
    return pos.numpy()


def modal_predict(model, s, ep, loads):
    t = torch.from_numpy(loads)
    n = len(t)

    def expand(key):
        src = s[key][0] if key in ("q", "v") else s[key]
        return torch.from_numpy(src).expand(n, *src.shape)

    pos = model(expand("q"), expand("v"), expand("static"), t).numpy()
    return np.stack(
        [
            decode(row, s["static"], ep["rest_position"], ep["inverse_mass"] > 0)
            for row in pos
        ]
    )


def case(task):
    seed, paths, out, cpu = task
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    teacher = Teacher(seed)
    ep = teacher.episode
    meta = dict(
        ep["metadata"],
        family="cable",
        seed=seed,
        identity=f"cable-{seed}",
        excitation="gate",
        sha256="",
    )
    s = pack(meta, ep)
    # Only rest positions, initial q/v, geometry/material descriptors and the prescribed inputs enter a planner.
    with torch.inference_mode():
        actors = {}
        for key in ("supernerd", "supernerd_mixed"):
            net = ModalForward().eval()
            net.load_state_dict(
                torch.load(paths[key], map_location="cpu", weights_only=True)
            )
            actors[key] = (modal_predict, net)
        bundle = torch.load(paths["neurall"], map_location="cpu", weights_only=True)
        neurall = PortableRecurrent(bundle).eval()
        actors["neurall"] = (
            lambda model, s, ep, u: neurall_predict(bundle, model, s, ep, u),
            neurall,
        )

        # Every model emits the entire native pose record. Separate timing avoids counting scores from the targets.
        def fast_bank(actor):
            start = perf_counter()
            loads = np.broadcast_to(s["u"], (len(COMMANDS), *s["u"].shape)).copy()
            loads[:, :, 1] += COMMANDS @ teacher.shapes[:-1].T
            fn, net = actor
            pred = fn(net, s, ep, loads)
            gates = pred[:, GATES - 1, -1, 1].copy()
            elapsed = perf_counter() - start
            return pred, gates, elapsed

        catalog = {}
        methods = {}
        for key, actor in actors.items():
            fast_bank(actor)
            clocks = []
            for _ in range(5):
                pred, gate, elapsed = fast_bank(actor)
                clocks.append(elapsed)
            catalog[key] = pred
            methods[key] = {
                "gate": gate,
                "time": float(np.median(clocks)),
                "blocks_s": clocks,
            }
        start = perf_counter()
        stock = np.stack([teacher.run(a) for a in COMMANDS])
        newton_gate = stock[:, GATES, -1, 1].copy()
        newton_time = perf_counter() - start
        methods["newton"] = {
            "gate": newton_gate,
            "time": newton_time,
            "blocks_s": [newton_time],
        }
        methods["zero"] = {"gate": newton_gate, "time": 0.0, "blocks_s": []}
    idx_zero = int(np.flatnonzero(np.all(COMMANDS == 0, axis=1))[0])
    np.testing.assert_array_equal(stock[idx_zero], ep["position"])
    actions = targets(seed)
    examples = {
        "times": np.arange(FRAMES) * DT,
        "commands": COMMANDS,
        "rest": ep["rest_position"],
    }
    rows = []
    for i, a in enumerate(actions):
        # The oracle's continuous witness is revealed to the scorer only.
        original = teacher.run(a)
        goal = original[GATES, -1, 1].copy()
        row = {
            "seed": seed,
            "target": i,
            "gate_y_m": goal.tolist(),
            "witness": a.tolist(),
            "methods": {},
        }
        if i == 0:
            examples["target"] = goal
        for key, stats in methods.items():
            st = perf_counter()
            chosen = (
                int(np.argmin(np.abs(stats["gate"] - goal).max(axis=1)))
                if key != "zero"
                else idx_zero
            )
            pick = perf_counter() - st
            st = perf_counter()
            actual = teacher.run(COMMANDS[chosen])
            execute = perf_counter() - st
            np.testing.assert_array_equal(actual, stock[chosen])
            refined = teacher.run(COMMANDS[chosen], refined=True)
            delta = np.abs(actual[GATES, -1, 1] - goal)
            ref_delta = np.abs(refined[GATES, -1, 1] - goal)
            info = {
                "candidate": chosen,
                "amplitudes_m_s2": COMMANDS[chosen].tolist(),
                "predicted_gate_distance_mm": (
                    1000 * np.abs(stats["gate"][chosen] - goal)
                ).tolist(),
                "actual_gate_distance_mm": (1000 * delta).tolist(),
                "refined_gate_distance_mm": (1000 * ref_delta).tolist(),
                "success": bool(delta.max() <= TOLERANCE),
                "refined_success": bool(ref_delta.max() <= TOLERANCE),
                "planning_s": stats["time"] + pick,
                "physical_replay_s": execute,
                "plan_plus_execution_s": stats["time"] + pick + execute,
            }
            if key in catalog:
                displayed = catalog[key][chosen]
                nominal_lengths = np.linalg.norm(
                    np.diff(ep["rest_position"], axis=0), axis=-1
                )
                info["model_edge_strain_rms_pct"] = float(
                    100
                    * np.sqrt(
                        np.mean(
                            (
                                np.linalg.norm(np.diff(displayed, axis=1), axis=-1)
                                / nominal_lengths
                                - 1
                            )
                            ** 2
                        )
                    )
                )
                info["native_rms_mm"] = float(
                    1000
                    * np.sqrt(np.mean(np.sum((displayed - actual[1:]) ** 2, axis=-1)))
                )
                if i == 0:
                    examples[key] = np.concatenate([actual[:1], displayed])
                    examples[key + "_replay"] = actual
            row["methods"][key] = info
        rows.append(row)
    np.savez_compressed(out / f"example-{seed}.npz", **examples)
    receipt = {
        "cases": rows,
        "bank_clocks_s": {k: v["blocks_s"] for k, v in methods.items()},
        "initial_state_and_wrench_check_max_m": float(
            np.max(np.abs(stock[idx_zero] - ep["position"]))
        ),
    }
    (out / f"object-{seed}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return seed


def summarize(out):
    objects = [json.loads((out / f"object-{seed}.json").read_text()) for seed in SEEDS]
    cases = [r for obj in objects for r in obj["cases"]]
    results = {}
    for key in cases[0]["methods"]:
        rows = [r["methods"][key] for r in cases]
        distance = np.asarray([x["actual_gate_distance_mm"] for x in rows]).max(axis=1)
        results[key] = {
            "tasks": len(rows),
            "passed": sum(x["success"] for x in rows),
            "passed_refined": sum(x["refined_success"] for x in rows),
            "sensitivity_success_counts": {
                str(t): int((distance <= t).sum()) for t in (5, 10, 20)
            },
            "mean_max_gate_distance_mm": float(distance.mean()),
            "median_planning_ms": float(
                1000 * np.median([r["planning_s"] for r in rows])
            ),
            "median_plan_plus_execution_s": float(
                np.median([r["plan_plus_execution_s"] for r in rows])
            ),
        }
        if "native_rms_mm" in rows[0]:
            results[key]["native_rms_mm"] = float(
                np.mean([r["native_rms_mm"] for r in rows])
            )
            results[key]["edge_strain_rms_pct"] = float(
                np.mean([r["model_edge_strain_rms_pct"] for r in rows])
            )
    receipt = {
        "protocol": "Two lateral gates at 1.10 and 2.40 seconds, 10 mm tolerance at both instants. Same discrete 25 commands per planner, continuous hidden target witness. CPU-only. Each goal charged a fresh bank time; bank costs share a geometry with three other test goals.",
        "newton": {
            "substeps": 4,
            "iterations": 8,
            "refined_substeps": 8,
            "refined_iterations": 16,
        },
        "geometry_seeds": list(SEEDS),
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--mixed", type=Path, required=True)
    cli.add_argument("--neurall", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    paths = {
        "supernerd": args.parent,
        "supernerd_mixed": args.mixed,
        "neurall": args.neurall,
    }
    (args.output / "model-hashes.json").write_text(
        json.dumps(
            {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
            indent=2,
        )
        + "\n"
    )
    jobs = [
        (s, paths, args.output, cpus[i % args.workers])
        for i, s in enumerate(SEEDS)
        if not (args.output / f"object-{s}.json").exists()
    ]
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for seed in pool.map(case, jobs):
            print(seed, flush=True)
    summarize(args.output)


if __name__ == "__main__":
    main()
