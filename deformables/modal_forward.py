"""Fit and evaluate a force-driven neural state-space model of Newton cable and cloth."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ModalForward(nn.Module):
    """Stable complex modes initialized from the observed shape and velocity."""

    def __init__(self, width=96, modes=32):
        super().__init__()
        self.modes = modes
        self.register_buffer("static_mean", torch.zeros(12))
        self.register_buffer("static_std", torch.ones(12))
        self.register_buffer(
            "freq", torch.logspace(math.log10(0.12), math.log10(5.2), modes)
        )
        self.condition = nn.Sequential(
            nn.Linear(102, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.coefficients = nn.Linear(width, modes * 7)
        self.readout = nn.Sequential(
            nn.Linear(2 * modes + width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 45),
        )
        self.register_buffer(
            "moving", torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0] * 3).reshape(1, 1, 15, 1)
        )

    def initialize(self, q, v, static):
        normalized = (static - self.static_mean) / self.static_std
        context = self.condition(
            torch.cat((normalized, q.flatten(1), v.flatten(1) * 0.22), dim=1)
        )
        real, imag, shift, decay, bx, by, bz = (
            self.coefficients(context).unflatten(-1, (7, self.modes)).unbind(1)
        )
        frequency = self.freq * shift.tanh().exp() / static[:, 1:2].sqrt()
        pole = torch.polar(
            (-0.02 * (0.045 + torch.nn.functional.softplus(decay))).exp(),
            2 * math.pi * 0.02 * frequency,
        )
        state = torch.complex(real, imag)
        gains = torch.stack((bx, by, bz), dim=1)
        return state, pole, gains, context

    def decode(self, states, context):
        shared = context[:, None, :].expand(-1, states.shape[1], -1)
        return self.readout(torch.cat((states.real, states.imag, shared), dim=-1))

    def output_shape(self, delta, q, static):
        delta = delta.unflatten(-1, (15, 3)) * self.moving
        axial = delta.unflatten(-2, (3, 5)).mean(-3, keepdim=True)
        cable = axial.expand(-1, -1, 3, -1, -1).flatten(-3, -2)
        delta = torch.where(static[:, None, 0:1, None].bool(), delta, cable)
        return q[:, None] + delta

    def start(self, q, v, static):
        """Initialize a streaming carry from one observed deformable state."""
        state, pole, gains, context = self.initialize(q, v, static)
        origin = self.decode(state[:, None], context)
        return (q, static, pole, gains, context, origin), state

    def advance(self, fixed, state, control):
        """Apply one current prescribed acceleration and emit the next shape."""
        q, static, pole, gains, context, origin = fixed
        drive = ((control[:, None] / 9.81) @ gains).squeeze(1)
        state = pole * state + (1 - pole) * drive
        position = self.output_shape(
            self.decode(state[:, None], context) - origin, q, static
        )
        return state, position[:, 0]

    def forward(self, q, v, static, controls, *, streaming=False):
        state, pole, gains, context = self.initialize(q, v, static)
        origin = self.decode(state[:, None], context)
        step_input = (controls / 9.81) @ gains
        if streaming:
            history = []
            for force in step_input.unbind(dim=1):
                state = pole * state + (1 - pole) * force
                history.append(state)
            states = torch.stack(history, dim=1)
        else:
            # Evaluate exactly the same causal recurrence for an entire supplied control sequence.
            h = controls.shape[1]
            powers = pole[:, None] ** torch.arange(h, device=pole.device)[None, :, None]
            fft_length = 1 << (2 * h - 2).bit_length()
            kernel = powers * (1 - pole[:, None])
            driven = torch.fft.ifft(
                torch.fft.fft(step_input, n=fft_length, dim=1)
                * torch.fft.fft(kernel, n=fft_length, dim=1),
                dim=1,
            )[:, :h]
            states = driven + powers * pole[:, None] * state[:, None]
        return self.output_shape(self.decode(states, context) - origin, q, static)


def load(path, device, limit=None):
    with np.load(path) as f:
        families = f["family"]
        indices = np.concatenate(
            [
                np.flatnonzero(families == family)[:limit]
                for family in ("cable", "cloth")
            ]
        )
        return {
            k: torch.as_tensor(f[k][indices], dtype=torch.float32, device=device)
            if k in {"q", "v", "static", "u", "length"}
            else f[k][indices]
            for k in f.files
        }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def metric(truth, pred, length):
    rms = (pred - truth).square().sum(dim=-1).mean(dim=-1).sqrt()
    return (
        (100 * rms.mean(dim=1)).tolist(),
        (1000 * length * rms.mean(dim=1)).tolist(),
        (100 * rms[:, -1]).tolist(),
    )


@torch.no_grad()
def evaluate(model, data, start, keep_predictions=False):
    q, v, s, u, lengths = (data[x] for x in ("q", "v", "static", "u", "length"))
    pred = model(q[:, start], v[:, start], s, u[:, start:])
    truth = q[:, start + 1 :]
    algorithms = {"model": pred, "persistence": q[:, start, None].expand_as(truth)}
    result = {}
    for label, prediction in algorithms.items():
        pct, mm, terminal = metric(truth, prediction, lengths)
        result[label] = {}
        for family in ("cable", "cloth"):
            use = data["family"] == family
            result[label][family] = {
                "cases": int(use.sum()),
                "mean_length_pct": float(np.mean(np.array(pct)[use])),
                "mean_mm": float(np.mean(np.array(mm)[use])),
                "final_length_pct": float(np.mean(np.array(terminal)[use])),
                "case_length_pct": np.array(pct)[use].tolist(),
            }
    if keep_predictions:
        errors = {"cable": [], "cloth": []}
        for family, values in errors.items():
            for seed in np.unique(data["seed"][data["family"] == family]):
                family_seed = (data["family"] == family) & (data["seed"] == seed)
                references = np.flatnonzero(
                    family_seed & (data["excitation"] == "reference")
                )
                if not len(references):
                    continue
                ref = int(references[0])
                for i in np.flatnonzero(
                    family_seed & (data["excitation"] != "reference")
                ):
                    signal = truth[i] - truth[ref]
                    resid = (pred[i] - pred[ref]) - signal
                    values.append(
                        float(
                            100
                            * resid.square().mean().sqrt()
                            / signal.square().mean().sqrt()
                        )
                    )
        result["paired_added_motion"] = {
            k: {
                "case_relative_pct": values,
                "mean_relative_pct": float(np.mean(values)),
            }
            for k, values in errors.items()
            if values
        }
    return result, pred


def mean_dev(model, data):
    return {
        str(s): {
            f: v["mean_length_pct"]
            for f, v in evaluate(model, data, s)[0]["model"].items()
        }
        for s in (0, 100)
    }


def fit(data, dev, updates, batch_size, seed, device):
    torch.manual_seed(seed)
    model = ModalForward().to(device)
    model.static_mean.copy_(data["static"].mean(dim=0))
    model.static_std.copy_(data["static"].std(dim=0).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, fused=device.type == "cuda"
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, updates, eta_min=2e-4
    )
    sync(device)
    start = time.perf_counter()
    excluded_eval = 0.0
    curve = []
    for step in range(1, updates + 1):
        horizon = 100 if step <= updates // 3 else 200
        n = len(data["q"])
        rows = torch.randint(n, (batch_size,), device=device)
        starts = torch.randint(301 - horizon, (batch_size,), device=device)
        starts[: batch_size // 4] = 0
        t = starts[:, None] + torch.arange(horizon, device=device)
        q = data["q"][rows, starts]
        v = data["v"][rows, starts]
        pos = model(q, v, data["static"][rows], data["u"][rows[:, None], t])
        err = pos - data["q"][rows[:, None], t + 1]
        velocity_error = err[:, 1:] - err[:, :-1]
        loss = (err.square().mean(dim=(-1, -2)) + 1e-7).sqrt().mean()
        loss = (
            loss
            + 0.7 * (velocity_error.square().mean(dim=(-1, -2)) + 1e-8).sqrt().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        schedule.step()
        if step in {1, updates // 8, updates // 4, updates // 2, updates}:
            sync(device)
            checkpoint = time.perf_counter()
            curve.append(
                {
                    "update": step,
                    "fit_s": checkpoint - start - excluded_eval,
                    "dev": mean_dev(model, dev),
                }
            )
            sync(device)
            excluded_eval += time.perf_counter() - checkpoint
    return model.eval(), curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("modal"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 32, 192])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dev = load(args.data / "development.npz", device)
    test = load(args.data / "test.npz", device)
    report = {
        "model": "Stable force-driven neural state-space model",
        "seed": 91,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "budgets": {},
        "train_identity_order": "First n ascending seeds for each family",
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "hardware": torch.cuda.get_device_name()
        if device.type == "cuda"
        else platform.processor(),
        "scores": "Mean of per-frame RMS Euclidean point error on 15 canonical nodes divided by rest length, across cases",
    }
    for budget in args.budgets:
        start = time.perf_counter()
        data = load(args.data / "train.npz", device, budget)
        sync(device)
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        model, curve = fit(data, dev, args.updates, args.batch_size, 91, device)
        sync(device)
        full_fit_s = time.perf_counter() - start
        info = {
            "curve": curve,
            "data_load_s": load_s,
            "full_fit_with_dev_s": full_fit_s,
            "params": sum(p.numel() for p in model.parameters()),
            "eval": {},
        }
        for t in (0, 100):
            result, pred = evaluate(model, test, t, True)
            info["eval"][str(t)] = result
            if budget == max(args.budgets):
                examples = [
                    np.flatnonzero(
                        (test["family"] == f)
                        & (test["excitation"] == "descending-chirp")
                    )[0]
                    for f in ("cable", "cloth")
                ]
                np.savez_compressed(
                    args.output / f"examples-{t}.npz",
                    reference=test["q"][examples, t:].cpu().numpy(),
                    predicted=pred[examples].cpu().numpy(),
                    family=test["family"][examples],
                    identity=test["identity"][examples],
                    rest=test["rest"][examples],
                    length=test["length"][examples].cpu().numpy(),
                )
        with torch.no_grad():
            tiny = [test[k][:2] for k in ("q", "v", "static", "u")]
            regular = model(tiny[0][:, 0], tiny[1][:, 0], tiny[2], tiny[3])
            streamed = model(
                tiny[0][:, 0], tiny[1][:, 0], tiny[2], tiny[3], streaming=True
            )
            info["max_stream_difference_body_lengths"] = float(
                (regular - streamed).abs().max()
            )
        report["budgets"][str(budget)] = info
        torch.save(model.state_dict(), args.output / f"weights-{budget}.pt")
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            budget,
            curve[-1],
            {
                s: {f: x["mean_length_pct"] for f, x in z["model"].items()}
                for s, z in info["eval"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
