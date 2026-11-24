# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture Newton trajectories across training loads and paired held waveforms."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .dataset import SCHEMA, _write_episode
from .elastic_excitations import DT, FRAMES, ONSET

_PROGRAMS = ("train-pulses", "train-diverse", "train-wide", "held-composition")
_TRAIN_PROFILES = ("smooth-bursts", "fixed-multisine", "smooth-knots")
_WIDE_PROFILES = (
    "reference",
    "legacy-diverse",
    "legacy-diverse",
    "broad-pulse",
    "signed-bursts",
    "redirected-bursts",
    "continuous-multisine",
    "rotating-multisine",
    "irregular-knots",
    "sloped-drive",
)
_HELD_PROFILES = ("triangle-load", "descending-chirp")
_SPLIT_SEEDS = {
    "train": ((20_000, 30_000), (100_000, 1_100_000)),
    "development": ((30_000, 40_000), (1_100_000, 1_200_000)),
    "test": ((40_000, 50_000), (1_200_000, 1_300_000)),
}
_MAX_SLEW = 100.0


def _pulse(elapsed, width):
    inside = (elapsed > 0) & (elapsed < width)
    return np.where(inside, np.sin(np.pi * np.clip(elapsed / width, 0, 1)) ** 2, 0)


def _training_profile(program, seed, rng):
    pulse_width = float(rng.uniform(0.45, 1.10))
    if program == "train-pulses":
        return "single-pulse", pulse_width, lambda t: _pulse(t, pulse_width)

    profile = _TRAIN_PROFILES[seed % len(_TRAIN_PROFILES)]
    if profile == "smooth-bursts":
        count = rng.integers(2, 5)
        widths = rng.uniform(0.25, 0.45, count)
        gaps = rng.uniform(0.06, 0.16, count - 1)
        starts = np.r_[0.0, np.cumsum(widths[:-1] + gaps)]
        amplitudes = rng.uniform(0.35, 1.0, count) * (-1.0) ** np.arange(count)

        def bursts(t):
            return sum(z * _pulse(t - a, width) for z, a, width in zip(amplitudes, starts, widths, strict=True))

        return profile, float(starts[-1] + widths[-1]), bursts

    duration = float(rng.uniform(1.3, 2.2))
    if profile == "fixed-multisine":
        frequencies = rng.choice((0.55, 0.8, 1.2, 1.65, 2.15), 2, replace=False)
        phase = rng.uniform(0, 2 * np.pi, 2)
        mixing = rng.uniform(0.3, 0.7)

        def multisine(t):
            angles = 2 * np.pi * t[:, None] * frequencies + phase
            combined = np.sin(angles) @ np.array((mixing, 1 - mixing))
            return combined * _pulse(t, duration)

        return profile, duration, multisine

    knot_count = rng.integers(5, 8)
    values = np.r_[0.0, rng.uniform(-1.0, 1.0, knot_count - 2), 0.0]

    def knots(t):
        coordinate = np.clip(t / duration, 0, 1) * (knot_count - 1)
        left = np.minimum(coordinate.astype(int), knot_count - 2)
        fraction = coordinate - left
        weight = fraction**2 * (3.0 - 2.0 * fraction)
        return (1.0 - weight) * values[left] + weight * values[left + 1]

    return profile, duration, knots


def _held_profile(profile):
    if profile == "triangle-load":

        def triangle(t):
            return np.clip(np.minimum(t / 0.28, (1.20 - t) / 0.92), 0, 1)

        return 1.20, triangle

    duration = 1.65

    def chirp(t):
        elapsed = np.clip(t, 0, duration)
        ramp = np.clip(np.minimum(elapsed, duration - elapsed) / 0.24, 0, 1)
        cycles = 3.0 * elapsed + 0.5 * (0.65 - 3.0) / duration * elapsed**2
        return np.sin(2.0 * np.pi * cycles) * np.sin(np.pi * ramp / 2.0) ** 2

    return duration, chirp


def _match_sampled_load(time, profile, onset, duration, peak, direction):
    """Scale to the requested sampled vector peak and stretch time if needed."""
    stretch = 1.0
    for _ in range(8):
        if onset + duration * stretch > time[-1] - DT:
            raise ValueError("The slew-capped load exceeds the recorded horizon")
        scalar = profile((time - onset) / stretch)
        maximum = np.abs(scalar).max()
        if maximum <= 0:
            raise ValueError("The generated waveform is zero at every sampled frame")
        load = (scalar[:, None] * (peak / maximum) * direction).astype(np.float32)
        sampled_peak = float(np.linalg.norm(load, axis=1).max())
        sampled_slew = float(np.linalg.norm(np.diff(load, axis=0), axis=1).max() / DT)
        if sampled_slew <= _MAX_SLEW:
            return load, {
                "peak_added_acceleration_m_s2": sampled_peak,
                "peak_added_slew_m_s3": sampled_slew,
                "added_load_duration_s": duration * stretch,
                "added_load_time_stretch": stretch,
            }
        stretch *= 1.02 * sampled_slew / _MAX_SLEW
    raise ValueError("The added waveform did not converge to the sampled slew cap")


def _training_load(time, program, seed):
    if program == "train-wide":
        return _wide_load(time, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 40_153]))
    onset = float(rng.uniform(1.9, 2.4))
    peak = float(rng.uniform(3.0, 6.0))
    azimuth = float(rng.uniform(-np.pi, np.pi))
    direction = np.array((np.cos(azimuth), np.sin(azimuth), rng.uniform(-0.35, 0.35)))
    direction /= np.linalg.norm(direction)
    family, duration, sampler = _training_profile(program, seed, rng)
    added, statistics = _match_sampled_load(time, sampler, onset, duration, peak, direction)
    return added, {
        "excitation": family,
        "load_level_m_s2": peak,
        "added_load_onset_s": onset,
        "added_load_direction": direction.tolist(),
        **statistics,
    }


def _unit_directions(rng, count):
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _wide_profile(elapsed, duration, family, direction, rng):
    if family == "broad-pulse":
        return _pulse(elapsed, duration)[:, None] * direction

    if family in ("signed-bursts", "redirected-bursts"):
        count = rng.integers(2, 7)
        widths = rng.uniform(0.6, 1.2, count)
        gaps = rng.uniform(0.2, 0.85, count - 1)
        scale = duration / (widths.sum() + gaps.sum())
        starts = np.r_[0.0, np.cumsum(widths[:-1] + gaps)] * scale
        pulses = np.column_stack(
            [_pulse(elapsed - start, width * scale) for start, width in zip(starts, widths, strict=True)]
        )
        weights = rng.uniform(0.35, 1.0, count)
        directions = (
            _unit_directions(rng, count)
            if family == "redirected-bursts"
            else (-1.0) ** np.arange(count)[:, None] * direction
        )
        return pulses @ (weights[:, None] * directions)

    if family in ("continuous-multisine", "rotating-multisine"):
        count = rng.integers(2, 6)
        frequencies = np.exp(rng.uniform(np.log(0.30), np.log(4.2), count))
        phases = rng.uniform(-np.pi, np.pi, count)
        weights = rng.uniform(0.25, 1.0, count)
        directions = _unit_directions(rng, count) if family == "rotating-multisine" else direction
        angles = 2 * np.pi * elapsed[:, None] * frequencies + phases
        return _pulse(elapsed, duration)[:, None] * (np.sin(angles) @ (weights[:, None] * directions))

    t = np.clip(elapsed / duration, 0, 1)
    if family == "irregular-knots":
        count = rng.integers(4, 10)
        knots = np.r_[0.0, np.cumsum(rng.uniform(0.3, 2.0, count - 1))]
        knots /= knots[-1]
        values = np.zeros((count, 3))
        values[1:-1] = rng.uniform(0.3, 1.0, (count - 2, 1)) * _unit_directions(rng, count - 2)
        left = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, count - 2)
        fraction = (t - knots[left]) / (knots[left + 1] - knots[left])
        weight = (fraction**2 * (3 - 2 * fraction))[:, None]
        return (1 - weight) * values[left] + weight * values[left + 1]

    rise, fall = rng.uniform(0.15, 0.32, 2)
    ramp = np.clip(np.minimum(t / rise, (1 - t) / fall), 0, 1)
    envelope = ramp**2 * (3 - 2 * ramp)
    start, stop = rng.uniform(0.2, 1.0, 2)
    return (envelope * ((1 - t) * start + t * stop))[:, None] * direction


def _wide_load(time, seed):
    """Draw a reproducible added load with bounded sampled peak and slew."""
    family = _WIDE_PROFILES[seed % len(_WIDE_PROFILES)]
    if family == "reference":
        return None, {
            "excitation": family,
            "load_level_m_s2": 0.0,
            "peak_added_acceleration_m_s2": 0.0,
            "peak_added_slew_m_s3": 0.0,
        }
    if family == "legacy-diverse":
        return _training_load(time, "train-diverse", seed)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 44_521]))
    duration_bounds = (0.32, 1.50) if family == "broad-pulse" else (1.05, 3.65)
    duration = float(rng.uniform(*duration_bounds))
    quiet_tail = rng.uniform(0.35, 1.2)
    onset = float(rng.uniform(0.35, time[-1] - DT - quiet_tail - duration))
    requested_peak = float(rng.uniform(0.5, 11.0))
    direction = _unit_directions(rng, 1)[0]
    if rng.random() < 0.25:
        direction = rng.choice((-1.0, 1.0)) * np.eye(3)[rng.integers(3)]

    raw = _wide_profile(time - onset, duration, family, direction, rng)
    norms = np.linalg.norm(raw, axis=1)
    peak_i = np.argmax(norms)
    raw_peak = norms[peak_i]
    raw_slew = np.linalg.norm(np.diff(raw, axis=0), axis=1).max() / DT
    if not (np.isfinite(raw_peak) and np.isfinite(raw_slew) and raw_peak > 0 and raw_slew > 0):
        raise ValueError(f"Invalid sampled training waveform for scene {seed}")
    # Limit amplitude before casting so even fast, irregular drives keep their start and rest intervals.
    scale = min(requested_peak / raw_peak, 0.999 * _MAX_SLEW / raw_slew)
    load = (scale * raw).astype(np.float32)
    sampled_peak = float(np.linalg.norm(load, axis=1).max())
    sampled_slew = float(np.linalg.norm(np.diff(load, axis=0), axis=1).max() / DT)
    return load, {
        "excitation": family,
        "load_level_m_s2": sampled_peak,
        "requested_added_acceleration_m_s2": requested_peak,
        "added_load_onset_s": onset,
        "added_load_direction": (raw[peak_i] / raw_peak).tolist(),
        "peak_added_acceleration_m_s2": sampled_peak,
        "peak_added_slew_m_s3": sampled_slew,
        "added_load_duration_s": duration,
        "added_load_time_stretch": 1.0,
    }


def _held_load(time, profile, peak):
    direction = np.array((0.0, 1.0, 0.0))
    metadata = {"excitation": profile, "load_level_m_s2": peak}
    if profile == "reference":
        return None, metadata | {"peak_added_acceleration_m_s2": 0.0, "peak_added_slew_m_s3": 0.0}
    duration, sampler = _held_profile(profile)
    added, statistics = _match_sampled_load(time, sampler, ONSET, duration, peak, direction)
    return added, metadata | {"added_load_onset_s": ONSET, "added_load_direction": direction.tolist()} | statistics


def _generate(task):
    from .teachers import generate_episode  # noqa: PLC0415 -- initialize Warp inside the worker

    output, program, split, material, seed, profile, peak = task
    time = np.arange(FRAMES, dtype=np.float64) * DT
    if program == "held-composition":
        added, metadata = _held_load(time, profile, peak)
        tag = f"{profile}-{peak}" if peak else profile
    else:
        added, metadata = _training_load(time, program, seed)
        tag = program
    identity = f"{material}-{seed}-{tag}"
    episode = generate_episode(material, seed, frames=FRAMES, dt=DT, device="cpu", additional_load=added)
    episode["metadata"].update(metadata, load_diversity_program=program)
    return _write_episode(output, identity, material, split, episode, "Newton SolverVBD, load-diversity comparison")


def main():
    """Write one controlled training arm or the separate paired holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--cases", required=True, type=int, help="Distinct geometry seeds per material")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--program", required=True, choices=_PROGRAMS)
    parser.add_argument("--split", choices=("train", "development", "test"))
    args = parser.parse_args()
    held = args.program == "held-composition"
    split = args.split or ("test" if held else "train")
    if held != (split == "test"):
        parser.error("Reserve the held composition for test and other programs for train or development")
    end = args.seed + args.cases
    if (
        args.workers < 1
        or args.cases < 1
        or not any(start <= args.seed < end <= stop for start, stop in _SPLIT_SEEDS[split])
    ):
        parser.error("Choose positive cases and workers and a seed range reserved for the split")
    args.output.mkdir(parents=True, exist_ok=False)
    conditions = [(None, None)]
    if held:
        conditions = [("reference", 0)] + [(name, level) for name in _HELD_PROFILES for level in (3, 6)]
    tasks = [
        (args.output, args.program, split, material, seed, profile, peak)
        for material in ("cloth", "cable")
        for seed in range(args.seed, args.seed + args.cases)
        for profile, peak in conditions
    ]
    records = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for record in pool.map(_generate, tasks):
            records.append(record)
            print(record["identity"], flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "episodes": records}, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
