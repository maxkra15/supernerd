"""Check causality, closed-loop/parallel parity and CPU streaming latency."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from modal_forward import ModalForward, load


def stream(model, q, v, static, controls):
    fixed, state = model.start(q, v, static)
    predictions = []
    for control in controls.unbind(dim=1):
        state, position = model.advance(fixed, state, control)
        predictions.append(position)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/test.npz"))
    parser.add_argument(
        "--weights", type=Path, default=Path("modal-long/weights-192.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("verify.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = ModalForward().eval()
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    data = load(args.data, "cpu")
    result = {
        "device": platform.processor() or platform.machine(),
        "cpu_threads": 1,
        "timing": {},
    }
    for batch in (1, 16):
        inputs = [
            data["q"][:batch, 100],
            data["v"][:batch, 100],
            data["static"][:batch],
            data["u"][:batch, 100:],
        ]
        pred = model(*inputs)
        ref = stream(model, *inputs)
        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            stream(model, *inputs)
            latencies.append((time.perf_counter() - start) * 1000 / inputs[-1].shape[1])
        result["timing"][str(batch)] = {
            "mean_ms_per_step_by_group": latencies,
            "median_ms_per_step": float(np.median(latencies)),
            "max_parallel_difference_float32": float((pred - ref).abs().max()),
        }
        assert torch.allclose(pred, ref, atol=2e-5, rtol=2e-5)
    # Verify independence of the first outputs from later input changes.
    model = model.double()
    inputs = [item.double() for item in inputs]
    full = model(*inputs)
    altered = inputs[-1].clone()
    altered[:, 60:] += 30
    delta = (full[:, :60] - model(*inputs[:-1], altered)[:, :60]).abs().max().item()
    result["future_control_change_max_abs_earlier_output_f64"] = delta
    assert delta < 1e-10
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
