"""Fit shared streaming cable and cloth dynamics and evaluate open-loop geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

DT = 0.02
NODES = 15
D = NODES * 3
SEED = 2841


class TemporalBlock(nn.Module):
    """Multi-query temporal attention and a gated MLP share one packed projection."""

    def __init__(self, width: int, heads: int = 2, head_dim: int = 12, span: int = 3):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.span = span
        self.norm = nn.LayerNorm(width)
        self.pack = nn.Linear(width, (heads + 2) * head_dim + width * 2)
        self.output = nn.Linear(heads * head_dim + width, width)
        self.relative_bias = nn.Parameter(torch.zeros(heads, span))

    def project(self, x):
        head = self.head_dim
        q, k, v, gate, linear = self.pack(self.norm(x)).split(
            [self.heads * head, head, head, x.shape[-1], x.shape[-1]], dim=-1
        )
        q = q.unflatten(-1, (self.heads, head))
        return q, k, v, torch.nn.functional.silu(gate) * linear

    def sequence(self, x):
        q, k, v, ff = self.project(x)
        # Collect keys in lag order, masking padding for cold starts.
        length = x.shape[1]
        clock = torch.arange(length, device=x.device)
        lag = torch.arange(self.span, device=x.device)
        index = clock[:, None] - lag
        keys, values = k[:, index.clamp_min(0)], v[:, index.clamp_min(0)]
        logits = torch.einsum("bthd,btld->bthl", q, keys) * (self.head_dim ** -0.5)
        logits = (logits + self.relative_bias).masked_fill(index[None, :, None] < 0, -torch.inf)
        heads = torch.einsum("bthl,btld->bthd", logits.softmax(-1), values)
        return x + self.output(torch.cat([heads.flatten(-2), ff], -1))

    def step(self, x, cache):
        q, k, v, ff = self.project(x)
        k, v = k[:, None], v[:, None]
        if cache is not None:
            k = torch.cat((k, cache[0]), dim=1)
            v = torch.cat((v, cache[1]), dim=1)
        logits = torch.einsum("bhd,bld->bhl", q, k) * (self.head_dim ** -0.5)
        weights = (logits + self.relative_bias[:, :k.shape[1]]).softmax(-1)
        heads = torch.einsum("bhl,bld->bhd", weights, v)
        result = x + self.output(torch.cat((heads.flatten(1), ff), -1))
        return result, (k[:, :self.span - 1], v[:, :self.span - 1])


class StreamingDynamics(nn.Module):
    """One parameter set for either modality with a bounded per-layer temporal cache."""

    def __init__(self, width: int = 48):
        super().__init__()
        features = D * 2 + 3 + 12
        self.embed = nn.Linear(features, width)
        self.blocks = nn.ModuleList([TemporalBlock(width), TemporalBlock(width)])
        self.out = nn.Linear(width, D * 2)
        self.direct = nn.Linear(features, D * 2, bias=False)
        nn.init.zeros_(self.direct.weight)
        nn.init.normal_(self.out.weight, std=0.004)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("innovation_scale", torch.ones(D * 2))
        clamp = torch.ones(NODES, 3)
        clamp[::5] = 0
        self.register_buffer("free", clamp.reshape(D))

    def input(self, s, u, prop):
        return (torch.cat([s, u, prop], dim=-1) - self.center) / self.scale

    def integrate(self, x, state, hidden, prop):
        q, v = state.split(D, dim=-1)
        dq, dv = ((self.out(hidden) + self.direct(x)) * self.innovation_scale).split(D, dim=-1)
        next_v = (0.9 * v + dv) * self.free
        next_q = (q + 0.5 * DT * (v + next_v) + dq) * self.free
        # The cable is represented by three repeated samples across its zero width.
        next_state = torch.stack([next_q, next_v], dim=-2)
        grid = next_state.unflatten(-1, (3, 5, 3))
        cable = grid.mean(dim=-3, keepdim=True)
        grid = cable + prop[..., 0, None, None, None, None] * (grid - cable)
        return grid.flatten(-4)

    def step(self, state, u, prop, caches):
        x = self.input(state, u, prop)
        z = self.embed(x)
        carry = []
        for block, cache in zip(self.blocks, caches, strict=True):
            z, updated = block.step(z, cache)
            carry.append(updated)
        return self.integrate(x, state, z, prop), carry

    def sequence(self, state, u, prop):
        x = self.input(state, u, prop)
        z = self.embed(x)
        for block in self.blocks:
            z = block.sequence(z)
        return self.integrate(x, state, z, prop)


def load(path, device, per_family=None):
    with np.load(path) as archive:
        subsets = []
        for family in ("cable", "cloth"):
            indices = np.flatnonzero(archive["family"] == family)
            indices = indices[np.argsort(archive["seed"][indices], kind="stable")]
            subsets.extend(indices[:per_family] if per_family else indices)
        ids = np.asarray(subsets) if per_family is not None else np.arange(len(archive["q"]))
        data = {key: archive[key][ids] for key in archive.files}
    for key in ("q", "v", "u", "static"):
        data[key] = torch.tensor(data[key], device=device)
    data["state"] = torch.cat((data.pop("q").flatten(2), data.pop("v").flatten(2)), -1)
    return data


@torch.no_grad()
def normalize(model, data):
    s, u, prop = data["state"], data["u"], data["static"]
    x = torch.cat((s[:, :-1], u, prop[:, None].expand(-1, u.shape[1], -1)), -1)
    model.center.copy_(x.mean((0, 1)))
    model.scale.copy_(x.std((0, 1)).clamp_min(1e-3))
    q, v = s.split(D, dim=-1)
    dq = q[:, 1:] - q[:, :-1] - 0.5 * DT * (v[:, 1:] + v[:, :-1])
    dv = v[:, 1:] - 0.9 * v[:, :-1]
    model.innovation_scale.copy_(torch.cat((dq.std((0, 1)).clamp_min(.00015),
                                             dv.std((0, 1)).clamp_min(.012)), -1))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, data, save=False):
    result, saved = {}, {}
    for start in (0, 100):
        current = data["state"][:, start]
        cache = [None] * len(model.blocks)
        predictions = []
        for t in range(start, data["u"].shape[1]):
            current, cache = model.step(current, data["u"][:, t], data["static"], cache)
            predictions.append(current[:, :D])
        predicted = torch.stack(predictions, 1).unflatten(-1, (NODES, 3))
        truth = data["state"][:, start + 1:, :D].unflatten(-1, (NODES, 3))
        error = (predicted - truth).square().sum(-1).mean(-1).sqrt()
        frame_means = error.mean(1).cpu().numpy()
        if not np.isfinite(frame_means).all():
            raise ValueError("Nonfinite autoregressive prediction")
        point = {}
        for family in ("cloth", "cable"):
            mask = data["family"] == family
            point[family] = {
                "pct_length": float(100 * frame_means[mask].mean()),
                "mm": float(1000 * (frame_means[mask] * data["length"][mask]).mean()),
                "cases": int(mask.sum()),
            }
        if save:
            point["cases"] = [
                {"identity": str(id_), "family": str(fam), "excitation": str(ex),
                 "pct_length": float(100 * score), "mm": float(1000 * score * length)}
                for id_, fam, ex, score, length in zip(
                    data["identity"], data["family"], data["excitation"], frame_means, data["length"], strict=True
                )
            ]
            # Preserve the observed starting frame for convenient replay rendering.
            initial = data["state"][:, start:start + 1, :D].unflatten(-1, (NODES, 3))
            saved[f"q{start}"] = torch.cat((initial, predicted), dim=1).cpu().numpy()
        result[str(start)] = point
    return result, saved


def train_batch(model, data, optimizer, batch, recursive, recursive_steps):
    s, u, prop = data["state"], data["u"], data["static"]
    n, frames = u.shape[:2]
    horizon = recursive_steps if recursive else 10
    i = torch.randint(n, (batch,), device=s.device)
    t = torch.randint(frames - horizon + 1, (batch,), device=s.device)
    times = t[:, None] + torch.arange(horizon, device=s.device)
    state, force = s[i[:, None], times], u[i[:, None], times]
    target = s[i[:, None], times + 1]
    static = prop[i]
    if recursive:
        current = state[:, 0]
        cache = [None] * len(model.blocks)
        rows = []
        for j in range(horizon):
            current, cache = model.step(current, force[:, j], static, cache)
            rows.append(current)
        predicted = torch.stack(rows, 1)
    else:
        predicted = model.sequence(state, force, static[:, None].expand(-1, horizon, -1))
    q_err, v_err = (predicted - target).split(D, dim=-1)
    q_scale = (model.innovation_scale[:D] + DT * model.innovation_scale[D:]).clamp_min(.0005)
    loss = (q_err / q_scale).square().mean() * .25 + (v_err / model.innovation_scale[D:]).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.)
    optimizer.step()


def make_updates(model, data, optimizer, batch, recursive, recursive_steps, stage_updates):
    device = data["state"].device
    if device.type != "cuda" or stage_updates <= 3:
        return lambda: train_batch(model, data, optimizer, batch, recursive, recursive_steps), 0
    # Count the three eager warmup updates. Captured operations execute on replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            train_batch(model, data, optimizer, batch, recursive, recursive_steps)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        train_batch(model, data, optimizer, batch, recursive, recursive_steps)
    # Capture records work. The operations execute on graph replay.
    return graph.replay, 3


@torch.no_grad()
def parity(model, data):
    import copy
    reference = copy.deepcopy(model).cpu().double()
    states = data["state"][:4, 20:33].cpu().double()
    u = data["u"][:4, 20:33].cpu().double()
    prop = data["static"][:4].cpu().double()
    full = reference.sequence(states, u, prop[:, None].expand(-1, states.shape[1], -1))
    cache = [None] * len(reference.blocks)
    out = []
    for j in range(states.shape[1]):
        row, cache = reference.step(states[:, j], u[:, j], prop, cache)
        out.append(row)
    error = (full - torch.stack(out, 1)).abs().max().item()
    if error > 1e-8:
        raise ValueError(f"Cached/full parity failed with {error}")
    return error


def fit(args, budget, dev, test, device):
    synchronize(device)
    start = perf_counter()
    train = load(args.data / "train.npz", device, budget)
    torch.manual_seed(SEED)
    model = StreamingDynamics().to(device)
    normalize(model, train)
    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001, fused=device.type=="cuda",
                                  capturable=device.type=="cuda")
    synchronize(device)
    setup_seconds = perf_counter() - start
    records = []
    current_update = 0
    train_seconds = 0.
    best = math.inf
    split_at = 0 if args.resume_model else max(1, round(.6 * args.updates))
    milestones = sorted({min(v, args.updates) for v in (50, 200, 500, 1000, 2000, 4000, args.updates)} | {split_at})
    for recursive, limit in ((False, split_at), (True, args.updates)):
        if current_update == limit:
            continue
        started = perf_counter()
        update, warm_count = make_updates(model, train, optimizer, args.batch, recursive, args.recursive_steps, limit - current_update)
        current_update += warm_count
        for target_step in [i for i in milestones if current_update < i <= limit]:
            for _ in range(current_update, target_step):
                update()
            synchronize(device)
            train_seconds += perf_counter() - started
            current_update = target_step
            eval_start = perf_counter()
            scores, _ = evaluate(model, dev)
            average = np.mean([scores[start][family]["pct_length"] for start in scores for family in ("cable","cloth")])
            if average < best:
                best = average
                selected = current_update
                torch.save(model.state_dict(), args.output / f"stream-{budget}.pt")
            synchronize(device)
            records.append({"updates": current_update, "fit_seconds": train_seconds, "development": scores,
                            "validation_seconds": perf_counter()-eval_start})
            print(budget, current_update, round(train_seconds, 2), round(float(average), 3), flush=True)
            started = perf_counter()
    path = args.output / f"stream-{budget}.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    error = parity(model, dev)
    metrics, predictions = evaluate(model, test, save=True)
    if budget == max(args.budgets):
        np.savez_compressed(args.output / "stream-predictions.npz", **predictions, identity=test["identity"])
    return {"trajectories_per_family": budget, "preprocessing_and_setup_s": setup_seconds,
            "fit_s": train_seconds, "updates_selected": selected, "test": metrics,
            "parameters": sum(p.numel() for p in model.parameters()), "development_curve": records,
            "cached_full_max_abs_float64": error, "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=(8,32,192))
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=.0015)
    parser.add_argument("--recursive-steps", type=int, default=6)
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(*args.budgets, args.updates, args.batch, args.recursive_steps) < 1:
        parser.error("training budget, updates, batch and recursive steps must be positive")
    if args.resume_model and len(args.budgets) != 1:
        parser.error("supply one data budget when resuming a fitted checkpoint")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    dev = load(args.data/"development.npz", device)
    test = load(args.data/"test.npz", device)
    result = {"seed": SEED, "dt": DT, "batch": args.batch, "update_budget": args.updates,
              "architecture": "Two width-48 parallel multi-query attention and SwiGLU blocks, two query heads, one KV head, span 3 per block, 5-frame total receptive field; trapezoidal integration with learned state residual and direct affine readout. One shared cable/cloth checkpoint per budget.",
              "device": torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
              "lr": args.lr, "recursive_steps": args.recursive_steps,
              "resume_model_sha256": hashlib.sha256(args.resume_model.read_bytes()).hexdigest() if args.resume_model else None,
              "timing": "Synchronized wall time, excludes validation, teacher data generation, and initial model/preprocessing setup. CUDA graph warm-up/capture included. Hardware can be shared.",
              "selection": "Lowest development mean percent-length error across cable and cloth and both restart horizons.",
              "runs": []}
    for budget in args.budgets:
        result["runs"].append(fit(args, budget, dev, test, device))
        (args.output/"stream-results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
