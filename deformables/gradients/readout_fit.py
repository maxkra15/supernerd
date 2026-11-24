"""Adapt a shared causal readout using exact feature derivatives and Newton secants."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from gradients.local_fit import KEYS, evaluate, initial_model, load_fit
from modal_forward import ModalForward

RIDGE_FRACTION = 0.01
RESPONSE_AMPLITUDE = 3.0
BASIS = torch.tensor([[1, 1, 1], [1, -1, 0], [1, 1, -2]], dtype=torch.float64)
BASIS = BASIS / torch.linalg.vector_norm(BASIS, dim=1, keepdim=True)


class Features(ModalForward):
    """Use the identical latent forward dynamics and return centered readout features."""

    def decode(self, states, context):
        shared = context[:, None, :].expand(-1, states.shape[1], -1)
        values = torch.cat((states.real, states.imag, shared), dim=-1)
        return self.readout[:-1](values)

    def output_shape(self, delta, q, static):
        return delta


def project(values, mode):
    # Rotate the 3 lateral rows; output 0 is the length-weighted row mean.
    y = values.unflatten(-2, (3, 5))
    return torch.einsum("l,...lpc->...pc", BASIS[mode], y)[..., 1:, :].flatten(-2, -1)


def statistics(features, residuals, kinds):
    grams, crosses = [], []
    for mode in range(3):
        selected = torch.ones_like(kinds).bool() if mode == 0 else kinds.bool()
        x = features[selected].reshape(-1, 96).double()
        y = project(residuals[selected].double(), mode).reshape(-1, 12)
        grams.append(x.T @ x / len(x))
        crosses.append(x.T @ y / len(x))
    return torch.stack(grams), torch.stack(crosses)


def linear_output(model, feature, static):
    raw = torch.nn.functional.linear(feature, model.readout[-1].weight, None)
    zero = torch.zeros((len(static), 15, 3), dtype=raw.dtype)
    return model.output_shape(raw, zero, static)


def solve(parent, pos, tangent=None):
    g, rhs = pos
    regularizer = (
        RIDGE_FRACTION
        * g.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
        * torch.eye(96)
    )
    if tangent is not None:
        g = g + RESPONSE_AMPLITUDE**2 * tangent[0]
        rhs = rhs + RESPONSE_AMPLITUDE**2 * tangent[1]
    coef = torch.linalg.solve(g + regularizer, rhs)
    delta = torch.einsum("ml,mfpc->lpcf", BASIS, coef.reshape(3, 96, 4, 3))
    updated = initial_model(parent, "cpu")
    updated.readout[-1].weight.unflatten(0, (3, 5, 3))[:, 1:, :, :].add_(delta.float())
    return updated


@torch.no_grad()
def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    clock = time.perf_counter
    start = clock()
    train = load_fit(args.data, "cpu")
    data_s = clock() - start
    features = Features()
    model = initial_model(args.parent, "cpu").eval()
    features.load_state_dict(model.state_dict())
    q, v, static, u, direction, truth = [train[k] for k in KEYS]
    kind = static[:, 0].bool()
    levels = torch.tensor([-0.5, 0.0, 0.5])[None, :, None, None]
    perts = (u[:, None] + levels * direction[:, None]).flatten(0, 1)
    s, vv, prop = [x.repeat_interleave(3, 0) for x in (q, v, static)]
    # Warm up both fixed feature operators on one TRAIN program.
    features(q[:1], v[:1], static[:1], u[:1])
    torch.func.jvp(
        lambda a: features(q[:1], v[:1], static[:1], a), (u[:1],), (direction[:1],)
    )
    start = clock()
    feat = features(s, vv, prop, perts)
    base = linear_output(model, feat, prop) + s[:, None]
    value_gt = truth.flatten(0, 1)
    pos = statistics(feat, value_gt - base, kind.repeat_interleave(3))
    stats_time = clock() - start
    max_abs = (base[:6] - model(s[:6], vv[:6], prop[:6], perts[:6])).abs().max().item()
    assert max_abs < 1e-5
    start = clock()
    _, df = torch.func.jvp(lambda a: features(q, v, static, a), (u,), (direction,))
    teacher_slope = truth[:, 2] - truth[:, 0]
    base_slope = linear_output(model, df, static)
    jac = statistics(df, teacher_slope - base_slope, kind)
    jac_time = clock() - start
    probe = []
    for path in sorted((args.data / "probe").glob("*.npz")):
        with np.load(path) as f:
            probe.append(dict(f))
    result = {
        "ridge_fraction_of_mean_position_feature_gram_eigenvalue": RIDGE_FRACTION,
        "derivative_equivalent_amplitude_m_s2": RESPONSE_AMPLITUDE,
        "fit_objects": len(kind),
        "fit_programs": len(perts),
        "probe_objects": len(probe),
        "fit_load_seconds": data_s,
        "feature_and_position_statistics_seconds": stats_time,
        "extra_jvp_and_tangent_statistics_seconds": jac_time,
        "feature_reconstruction_max": max_abs,
        "parent": {"summary": evaluate(model, probe)["summary"]},
    }
    for name, additional in (("positions", None), ("positions_plus_jvp", jac)):
        start = clock()
        updated = solve(args.parent, pos, additional).eval()
        solve_time = clock() - start
        result[name] = {
            "closed_form_solve_and_instantiate_s": solve_time,
            "fit_wall_s": stats_time + solve_time + (jac_time if additional else 0),
            **evaluate(updated, probe),
        }
        torch.save(updated.state_dict(), args.output / f"{name}.pt")
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in ("parent", "positions", "positions_plus_jvp"):
        print(name, result[name]["summary"])


if __name__ == "__main__":
    run()
