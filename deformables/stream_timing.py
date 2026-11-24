"""Time equivalent steady-state streaming and full-window forward rollouts on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from stream_deformables import DT, StreamingDynamics, load


@torch.no_grad()
def measure(model, data, batch, groups, steps):
    device = data["state"].device
    indices = torch.arange(batch, device=device) % len(data["static"])
    states = data["state"][indices, 100:105].contiguous()
    controls = data["u"][indices, 100:105].contiguous()
    prop = data["static"][indices].contiguous()
    static_window = prop[:, None].expand(-1, states.shape[1], -1)
    next_action = controls[:, -1].contiguous()

    # Prime both models from the same four prior frames. For the timing loop,
    # the prescribed force and material state variables remain GPU-resident.
    carry = [None] * len(model.blocks)
    for j in range(states.shape[1] - 1):
        _, carry = model.step(states[:, j], controls[:, j], prop, carry)
    state = states[:, -1].clone()
    buffered = [(k.contiguous(), v.contiguous()) for k, v in carry]

    def stream():
        nxt, kv = model.step(state, next_action, prop, buffered)
        state.copy_(nxt)
        for (key, value), (next_key, next_value) in zip(buffered, kv, strict=True):
            key.copy_(next_key)
            value.copy_(next_value)

    def window():
        nxt = model.sequence(states, controls, static_window)[:, -1]
        states.copy_(torch.cat((states[:, 1:], nxt[:, None]), dim=1))
        controls.copy_(torch.cat((controls[:, 1:], next_action[:, None]), dim=1))

    for _ in range(3):
        stream()
        window()
    joined_error = float((state - states[:, -1]).abs().max())
    initial_state = state.clone()
    initial_carry = [(k.clone(), v.clone()) for k, v in buffered]
    initial_window = states.clone()
    initial_controls = controls.clone()
    timing = {}
    for name, step in (("cached", stream), ("full_window", window)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize(device)
        batches = []
        for _ in range(groups):
            state.copy_(initial_state)
            states.copy_(initial_window)
            controls.copy_(initial_controls)
            for (key, value), (initial_key, initial_value) in zip(buffered, initial_carry, strict=True):
                key.copy_(initial_key)
                value.copy_(initial_value)
            torch.cuda.synchronize(device)
            start = perf_counter()
            for _ in range(steps):
                graph.replay()
            torch.cuda.synchronize(device)
            batches.append((perf_counter() - start) * 1000 / steps)
            actual = state if name == "cached" else states[:, -1]
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(f"{name} produced a nonfinite state in the timed window")
        timing[name] = {"group_ms": batches, "median_ms_per_batched_step": median(batches)}
    timing["full_over_cached"] = timing["full_window"]["median_ms_per_batched_step"] / timing["cached"]["median_ms_per_batched_step"]
    timing["warm_three_step_parity_max_abs"] = joined_error
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 80, 512))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda")
    model = StreamingDynamics().to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    data = load(args.data / "development.npz", device)
    result = {
        "device": torch.cuda.get_device_name(device),
        "seconds_per_physics_step": DT,
        "frames_for_full_window": 5,
        "precision": "float32",
        "measurement": "Warmed CUDA graph replay, complete state-to-state transition and cache/window maintenance on GPU, prescribed constant force, excluding initialization, host transfer and collision. Five groups of 100 steps.",
        "batches": {},
    }
    for batch in args.batches:
        result["batches"][str(batch)] = measure(model, data, batch, args.groups, args.steps)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
