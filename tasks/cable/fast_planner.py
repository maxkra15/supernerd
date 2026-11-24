"""Evaluate native cable-tip gate coordinates directly with the released temporal models."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cable_gates import (
    COMMANDS,
    GATES,
    SEEDS,
    ModalForward,
    PortableRecurrent,
    Teacher,
    modal_predict,
    neurall_predict,
    pack,
)
from scipy.interpolate import CubicSpline


def fast_modal(model, s, ep, patterns):
    # The latent recurrence is affine in the waveform; the trained readout remains nonlinear.
    q = torch.from_numpy(s["q"][0:1])
    v = torch.from_numpy(s["v"][0:1])
    static = torch.from_numpy(s["static"][None])
    state, pole, gain, ctx = model.initialize(q, v, static)
    origin = model.decode(state[:, None], ctx)
    force = torch.zeros((3, len(s["u"]), 3))
    force[0] = torch.from_numpy(s["u"])
    force[1:, :, 1] = torch.from_numpy(patterns[:-1].T)
    injection = (force / 9.81) @ gain
    powers = pole[:, None] ** torch.arange(int(GATES[-1]))[None, :, None]
    cmd = torch.from_numpy(COMMANDS)
    result = []
    for g in GATES:
        response = (injection[:, :g] * powers[:, :g].flip(1)).sum(1) * (1 - pole)
        base = (powers[:, g - 1] * pole) * state + response[0]
        result.append(base + cmd[:, 0:1] * response[1] + cmd[:, 1:2] * response[2])
    latent = torch.stack(result, 1)
    decoded = model.decode(latent, ctx.expand(len(cmd), -1)) - origin
    delta = decoded.unflatten(-1, (3, 5, 3))[..., -1, 1].mean(-1) + q.unflatten(
        1, (3, 5)
    )[:, :, -1, 1].mean(1)
    return ep["rest_position"][-1, 1] + s["length"] * delta.numpy()


def fast_neurall(bundle, model, s, ep, patterns):
    static = (torch.from_numpy(s["static"]) - bundle["center"]) / bundle["scale"]
    load = torch.from_numpy(s["u"]).repeat(len(COMMANDS), 1, 1)
    load[:, :, 1] += torch.from_numpy(COMMANDS @ patterns[:-1].T)
    inputs = torch.cat((static.expand(*load.shape[:2], -1), load / 10), -1)
    out = model.shape(model.core(inputs)[0][:, GATES - 1])
    rest = torch.from_numpy(ep["rest_position"])
    n = len(rest) - 1
    decoder = CubicSpline(np.linspace(0, 1, 17), np.eye(17), axis=0)(
        np.linspace(0, 1, n)
    ).astype("f")
    D = np.pad(np.kron(decoder, np.eye(3, dtype=np.float32)), ((3, 0), (0, 0)))
    D[np.repeat(ep["inverse_mass"] == 0, 3)] = 0
    readout = torch.from_numpy(D) @ bundle["basis"].T * s["length"]
    pos = rest + (out @ readout.T).unflatten(-1, (len(rest), 3))
    if not bundle["project_cable_length"]:
        return pos[:, :, -1, 1].numpy()
    ref = rest.diff(dim=0)
    delta = pos.diff(dim=2)
    norm = delta.norm(dim=-1, keepdim=True)
    direction = torch.where(
        norm > 1e-7, delta / norm.clamp_min(1e-7), ref / ref.norm(dim=-1, keepdim=True)
    )
    return (
        pos[:, :, 1, 1] + (direction[:, :, 1:, 1] * ref[1:].norm(dim=-1)).sum(2)
    ).numpy()


def select(gate, goal):
    return int(np.argmin(np.max(np.abs(gate - goal), axis=-1)))


def profile_geometry(seed, weights, nets, s, ep, scene, records):
    rows = []

    def full(name):
        controls = np.broadcast_to(s["u"], (len(COMMANDS), *s["u"].shape)).copy()
        controls[:, :, 1] += COMMANDS @ scene.shapes[:-1].T
        q = (
            neurall_predict(weights, nets[name], s, ep, controls)
            if name == "neurall"
            else modal_predict(nets[name], s, ep, controls)
        )
        return q[:, GATES - 1, -1, 1].copy()

    def fast(name):
        if name == "neurall":
            return fast_neurall(weights, nets[name], s, ep, scene.shapes)
        return fast_modal(nets[name], s, ep, scene.shapes)

    for name in nets:
        full_gate, fast_gate = full(name), fast(name)
        maxerr = float(np.max(np.abs(fast_gate - full_gate)))
        assert maxerr < 2e-5, (seed, name, maxerr)
        clocks = {"fast": [], "full": []}
        for method, func in (("full", full), ("fast", fast)):
            for rep in range(12):
                goal = np.asarray(records[rep % 4]["gate_y_m"])
                start = perf_counter()
                state = select(func(name), goal)
                clock = perf_counter() - start
                assert state == records[rep % 4]["methods"][name]["candidate"]
                clocks[method].append(clock)
        rows.append(
            {
                "seed": seed,
                "model": name,
                "gate_parity_max_m": maxerr,
                "decisions_matched": 4,
                **{
                    key + "_median_s": float(np.median(val))
                    for key, val in clocks.items()
                },
            }
        )
    return rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--mixed", type=Path, required=True)
    p.add_argument("--neurall", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--core", type=int, default=min(os.sched_getaffinity(0)))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.sched_setaffinity(0, {args.core})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    weights = torch.load(args.neurall, map_location="cpu", weights_only=True)
    nets = {"neurall": PortableRecurrent(weights).eval()}
    for name, path in (("supernerd", args.parent), ("supernerd_mixed", args.mixed)):
        nets[name] = ModalForward().eval()
        nets[name].load_state_dict(
            torch.load(path, weights_only=True, map_location="cpu")
        )
    rows = []
    for seed in SEEDS:
        rec = json.loads((args.results / f"object-{seed}.json").read_text())["cases"]
        scene = Teacher(seed)
        ep = scene.episode
        s = pack(
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

        rows.extend(profile_geometry(seed, weights, nets, s, ep, scene, rec))
    summary = {}
    for name in nets:
        r = [x for x in rows if x["model"] == name]
        summary[name] = {
            "full_ms": 1000 * float(np.median([x["full_median_s"] for x in r])),
            "fast_ms": 1000 * float(np.median([x["fast_median_s"] for x in r])),
            "gate_parity_max_m": max(x["gate_parity_max_m"] for x in r),
            "decisions_matched": sum(x["decisions_matched"] for x in r),
        }
    args.output.write_text(
        json.dumps(
            {"core": args.core, "summary": summary, "per_object": rows}, indent=2
        )
        + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
