# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless Newton teachers for cloth, cable, and coupled cloth-cable trajectories."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

__all__ = ["generate_episode"]

_GRAVITY = (0.0, 0.0, -9.81)
_SUBSTEPS = 4
_ITERATIONS = 8


def _load(time, rng, *, late_pulse=False):
    """Generate a smooth bounded load, held constant between recorded frames."""
    duration = float(rng.uniform(0.85, 1.65))
    phase = float(rng.uniform(-0.65, 0.65))
    amplitude = float(rng.uniform(2.0, 8.0))
    progress = np.clip(time / duration, 0.0, 1.0)
    envelope = np.sin(np.pi * progress) ** 2
    direction = np.stack(
        (
            0.35 * np.cos(np.pi * progress + phase),
            np.sin(2 * np.pi * progress + phase),
            -0.55 + 0.8 * np.sin(np.pi * progress),
        ),
        axis=-1,
    )
    acceleration = (amplitude * envelope[:, None] * direction).astype(np.float32)
    metadata = {
        "load_duration_seconds": duration,
        "load_phase": phase,
        "load_peak_acceleration_scale": amplitude,
    }
    if late_pulse:
        start = float(rng.uniform(1.85, 2.55))
        width = float(rng.uniform(0.45, 1.10))
        magnitude = float(rng.uniform(3.0, 9.0))
        azimuth = float(rng.uniform(-np.pi, np.pi))
        direction = np.array((np.cos(azimuth), np.sin(azimuth), rng.uniform(-0.5, 0.5)))
        active = (time >= start) & (time <= start + width)
        phase = (time[active] - start) / width
        acceleration[active] += (magnitude * np.sin(np.pi * phase[:, None]) ** 2 * direction).astype(np.float32)
        metadata.update(
            late_pulse_start_seconds=start,
            late_pulse_duration_seconds=width,
            late_pulse_acceleration_scale=magnitude,
            late_pulse_direction=direction.tolist(),
        )
    return acceleration, metadata


def _advance(first, second, control, solver, dt, force_array, force_field, substeps, collision_pipeline, contacts):
    max_contacts = 0
    for _ in range(substeps):
        first.clear_forces()
        getattr(first, force_field).assign(force_array)
        if contacts is not None:
            collision_pipeline.collide(first, contacts)
            max_contacts = max(max_contacts, int(contacts.soft_contact_count.numpy()[0]))
        solver.step(first, second, control, contacts, dt / substeps)
        first, second = second, first
    return first, second, max_contacts


def _record(model, solver, dt, values, force_array, decode, *, force_field, substeps, collision_pipeline=None):
    first, second = model.state(), model.state()
    control = model.control()
    contacts = collision_pipeline.contacts() if collision_pipeline is not None else None
    records = [decode(first)]
    contact_counts = [0]
    for value in values[:-1]:
        force_array.assign(value)
        first, second, max_contacts = _advance(
            first, second, control, solver, dt, force_array, force_field, substeps, collision_pipeline, contacts
        )
        records.append(decode(first))
        contact_counts.append(max_contacts)
    history = {name: np.stack([record[name] for record in records]) for name in records[0]}
    if contacts is not None:
        history["soft_contact_count"] = np.asarray(contact_counts, dtype=np.int32)
    return history


def _result(modality, seed, time, dt, records, rest, edges, kinds, inverse_mass, acceleration, metadata):
    state = {
        "time": time,
        **records,
        "rest_position": rest,
        "edge_index": edges.T.copy(),
        "edge_kind": kinds,
        "rest_length": np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1),
        "inverse_mass": inverse_mass,
        "gravity": np.asarray(_GRAVITY, dtype=np.float32),
        "external_acceleration": acceleration,
        "metadata": {
            "modality": modality,
            "seed": int(seed),
            "dt": dt,
            "substeps": _SUBSTEPS,
            "iterations": _ITERATIONS,
            "newton_version": newton.__version__,
            "warp_version": wp.__version__,
            **metadata,
        },
    }
    for name, value in state.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise FloatingPointError(f"Newton {modality} scene {seed} produced non-finite {name}")
    if np.abs(state["position"]).max() > 50 or np.linalg.norm(state["velocity"], axis=-1).max() > 60:
        raise FloatingPointError(f"Newton {modality} scene {seed} exceeded its bounded-motion domain")
    return state


def _cloth_topology(model):
    triangles = model.tri_indices.numpy().reshape(-1, 3).astype(np.int32)
    tri_pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]))
    structural = np.unique(np.sort(tri_pairs, axis=1), axis=0)
    return triangles, structural


def _cloth_edges(model, structural):
    # The first two entries of a Newton bending edge are opposite vertices.
    # Boundary edges have one opposing vertex missing and carry no hinge.
    opposite = model.edge_indices.numpy().reshape(-1, 4)[:, :2]
    opposite = np.sort(opposite[(opposite >= 0).all(axis=1)], axis=1).astype(np.int32)
    structural_pairs = set(map(tuple, structural))
    bend = np.asarray(sorted({tuple(edge) for edge in opposite}.difference(structural_pairs)), dtype=np.int32).reshape(
        -1, 2
    )
    return np.concatenate((structural, bend)), np.concatenate(
        (np.zeros(len(structural), dtype=np.int8), np.ones(len(bend), dtype=np.int8))
    )


def _cloth(
    seed,
    rng,
    time,
    dt,
    device,
    *,
    recorder=_record,
    late_pulse=False,
    additional_load=None,
    additional_load_at_middle=False,
):
    nx, ny = int(rng.integers(8, 13)), int(rng.integers(5, 9))
    length, width = float(rng.uniform(0.56, 0.83)), float(rng.uniform(0.32, 0.55))
    mass = float(rng.uniform(0.012, 0.027))
    stiffness = float(rng.uniform(900, 1900))
    damping = float(rng.uniform(0.8, 1.4))
    tilt = float(rng.uniform(-0.21, 0.21))
    builder = newton.ModelBuilder(gravity=_GRAVITY)
    builder.add_cloth_grid(
        pos=wp.vec3(-length / 2, -width / 2, 1.15),
        rot=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), tilt),
        vel=wp.vec3(0),
        dim_x=nx,
        dim_y=ny,
        cell_x=length / nx,
        cell_y=width / ny,
        mass=mass,
        fix_left=True,
        tri_ke=stiffness,
        tri_ka=stiffness,
        tri_kd=damping,
        edge_ke=0.08,
        edge_kd=0.002,
        particle_radius=0.008,
        label="neurall_cloth",
    )
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=_ITERATIONS)
    rest = model.particle_q.numpy().copy()
    inverse_mass = model.particle_inv_mass.numpy().copy()
    triangles, structural = _cloth_topology(model)
    edges, kinds = _cloth_edges(model, structural)

    load, controls = _load(time, rng, late_pulse=late_pulse)
    if additional_load is not None and not additional_load_at_middle:
        load += additional_load
    tip_edge = np.arange(nx, len(rest), nx + 1)
    acceleration = np.zeros((len(time), len(rest), 3), dtype=np.float32)
    acceleration[:, tip_edge] = load[:, None]
    particle_mass = model.particle_mass.numpy()
    force = acceleration * particle_mass[None, :, None]
    if additional_load_at_middle:
        middle_row = tip_edge - (nx - nx // 2)
        force[:, middle_row] = additional_load[:, None] * particle_mass[tip_edge][None, :, None]
        acceleration[:, middle_row] = force[:, middle_row] / particle_mass[middle_row][None, :, None]

    def decode(state):
        return {"position": state.particle_q.numpy().copy(), "velocity": state.particle_qd.numpy().copy()}

    values = recorder(
        model,
        solver,
        dt,
        force,
        wp.zeros(len(rest), dtype=wp.vec3, device=device),
        decode,
        force_field="particle_f",
        substeps=_SUBSTEPS,
    )
    result = _result(
        "cloth",
        seed,
        time,
        dt,
        values,
        rest,
        edges,
        kinds,
        inverse_mass,
        acceleration,
        {
            "solver": "SolverVBD",
            "cells": [nx, ny],
            "length_m": length,
            "width_m": width,
            "mass_per_free_vertex_kg": mass,
            "membrane_stiffness": stiffness,
            "membrane_damping": damping,
            "initial_tilt_rad": tilt,
            "control": "frame-held force distributed on the free edge; fixed opposite edge",
            **controls,
        },
    )
    result["triangles"] = triangles
    result["contact_radius"] = model.particle_radius.numpy().copy()
    return result


def _cable_topology(segment_count):
    segment_ids = np.arange(segment_count, dtype=np.int32)
    edges = np.concatenate(
        (
            np.stack((segment_ids, segment_ids + 1), axis=1),
            np.stack((segment_ids[:-1], segment_ids[:-1] + 2), axis=1),
        ),
    )
    kinds = np.concatenate((np.zeros(segment_count, dtype=np.int8), np.ones(segment_count - 1, dtype=np.int8)))
    return edges, kinds


def _cable_decoder(rest, bodies):
    """Read native segment COM state and export distal points plus the root start."""
    local = np.zeros((len(bodies), 3), dtype=np.float32)
    local[:, 2] = np.linalg.norm(np.diff(rest, axis=0), axis=1) * 0.5

    def decode(state):
        pose = state.body_q.numpy()[bodies]
        twist = state.body_qd.numpy()[bodies]
        center, quaternion = pose[:, :3], pose[:, 3:]
        axis = quaternion[:, :3]
        offset = local + 2 * np.cross(axis, np.cross(axis, local) + quaternion[:, 3:] * local)
        linear, angular = twist[:, :3], twist[:, 3:]
        endpoint_velocity = linear + np.cross(angular, offset)
        root_pos = center[:1] - offset[:1]
        root_vel = linear[:1] - np.cross(angular[:1], offset[:1])
        return {
            "position": np.concatenate((root_pos, center + offset)),
            "velocity": np.concatenate((root_vel, endpoint_velocity)),
            "segment_orientation": quaternion.copy(),
            "segment_center_velocity": linear.copy(),
            "segment_angular_velocity": angular.copy(),
        }

    return decode


def _cable(
    seed,
    rng,
    time,
    dt,
    device,
    *,
    recorder=_record,
    late_pulse=False,
    additional_load=None,
    additional_load_at_middle=False,
):
    segment_count = int(rng.integers(10, 16))
    length = float(rng.uniform(0.68, 1.02))
    radius = float(rng.uniform(0.0095, 0.014))
    density = float(rng.uniform(650.0, 920.0))
    stretch = float(rng.uniform(26000.0, 52000.0))
    bend = float(rng.uniform(7.0, 22.0))
    angle = float(rng.uniform(-0.15, 0.15))
    rod = newton.Rod.create_straight(
        start=wp.vec3(-length / 2, 0, 1.15),
        direction=wp.vec3(math.cos(angle), 0, math.sin(angle)),
        length=length,
        segment_count=segment_count,
        radius=radius,
    )
    builder = newton.ModelBuilder(gravity=_GRAVITY)
    cfg = builder.default_shape_cfg.copy()
    cfg.density = density
    bodies, _ = builder.add_rod(
        rod=rod,
        cfg=cfg,
        stretch_stiffness=stretch,
        stretch_damping=1.0,
        bend_stiffness=bend,
        bend_damping=0.25,
        twist_stiffness=bend * 0.8,
        twist_damping=0.2,
        label="neurall_cable",
    )
    root = bodies[0]
    builder.body_mass[root] = 0.0
    builder.body_inv_mass[root] = 0.0
    builder.body_inertia[root] = wp.mat33(0.0)
    builder.body_inv_inertia[root] = wp.mat33(0.0)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=_ITERATIONS)
    rest = np.asarray(rod.points, dtype=np.float32)
    edges, kinds = _cable_topology(segment_count)
    inverse_mass = np.concatenate((np.zeros(1, dtype=np.float32), model.body_inv_mass.numpy()[bodies]))
    load, controls = _load(time, rng, late_pulse=late_pulse)
    if additional_load is not None and not additional_load_at_middle:
        load += additional_load
    acceleration = np.zeros((len(time), len(rest), 3), dtype=np.float32)
    acceleration[:, -1] = load
    wrench = np.zeros((len(time), model.body_count, 6), dtype=np.float32)
    body_mass = model.body_mass.numpy()
    wrench[:, bodies[-1], :3] = load * body_mass[bodies[-1]]
    if additional_load_at_middle:
        middle = segment_count // 2
        body = bodies[middle]
        wrench[:, body, :3] = additional_load * body_mass[bodies[-1]]
        acceleration[:, middle + 1] = wrench[:, body, :3] / body_mass[body]
    force_buffer = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
    records = recorder(
        model,
        solver,
        dt,
        wrench,
        force_buffer,
        _cable_decoder(rest, bodies),
        force_field="body_f",
        substeps=_SUBSTEPS,
    )
    result = _result(
        "cable",
        seed,
        time,
        dt,
        records,
        rest,
        edges,
        kinds,
        inverse_mass,
        acceleration,
        {
            "solver": "SolverVBD",
            "segment_count": segment_count,
            "radius_m": radius,
            "length_m": length,
            "density_kg_m3": density,
            "stretch_stiffness_n_m": stretch,
            "bend_stiffness_nm_rad": bend,
            "initial_tilt_rad": angle,
            "control": "frame-held force at the final segment COM, encoded at its distal endpoint",
            "velocity": "endpoint from segment COM linear velocity and omega cross offset",
            **controls,
        },
    )
    result["contact_radius"] = np.full(len(rest), radius, dtype=np.float32)
    return result


def _mixed(seed, rng, time, dt, device, *, enable_contact):
    # The rectangle stays clear of the world floor. In the ablation run only
    # capsule-to-cloth collision is switched off, with identical initial data.
    nx, ny = 12, 10
    width, length = float(rng.uniform(0.55, 0.60)), float(rng.uniform(0.63, 0.76))
    height, cloth_radius = 0.84, 0.012
    cloth_mass = float(rng.uniform(0.0027, 0.0048))
    cloth_stiffness = float(rng.uniform(1600, 2400))
    cable_segments = int(rng.integers(8, 12))
    cable_length = float(rng.uniform(0.34, 0.42))
    cable_radius = float(rng.uniform(0.024, 0.029))
    cable_density = float(rng.uniform(620, 850))
    stretch = 38000.0
    bend = float(rng.uniform(2.4, 5.2))
    yaw = float(rng.uniform(-0.28, 0.28))
    pitch = float(rng.uniform(-0.12, 0.12))
    drop = float(rng.uniform(0.11, 0.19))
    direction = np.array((math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    center = np.array((float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.018, 0.018)), height + drop))
    start = center - 0.5 * cable_length * direction
    builder = newton.ModelBuilder(gravity=_GRAVITY)
    builder.add_cloth_grid(
        pos=wp.vec3(-length / 2, -width / 2, height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0),
        dim_x=nx,
        dim_y=ny,
        cell_x=length / nx,
        cell_y=width / ny,
        mass=cloth_mass,
        fix_left=True,
        fix_right=True,
        tri_ke=cloth_stiffness,
        tri_ka=cloth_stiffness,
        tri_kd=1.1,
        edge_ke=0.06,
        edge_kd=0.001,
        particle_radius=cloth_radius,
        label="neurall_coupled_cloth",
    )
    rod = newton.Rod.create_straight(
        start=wp.vec3(*start),
        direction=wp.vec3(*direction),
        length=cable_length,
        segment_count=cable_segments,
        radius=cable_radius,
    )
    cfg = builder.default_shape_cfg.copy()
    cfg.density = cable_density
    cfg.ke, cfg.kd, cfg.mu = 1.2e4, 0.8, 0.45
    cfg.has_shape_collision = False
    cfg.has_particle_collision = enable_contact
    bodies, _ = builder.add_rod(
        rod=rod,
        cfg=cfg,
        stretch_stiffness=stretch,
        stretch_damping=1.0,
        bend_stiffness=bend,
        bend_damping=0.2,
        twist_stiffness=bend * 0.8,
        twist_damping=0.16,
        label="neurall_coupled_cable",
    )
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.2e4
    model.soft_contact_kd = 0.8
    model.soft_contact_mu = 0.45
    solver = newton.solvers.SolverVBD(model, iterations=_ITERATIONS)
    pipeline = newton.CollisionPipeline(
        model,
        soft_contact_gap=0.009,
        soft_contact_max=2048,
        enable_rigid_soft_full_surface_contact=True,
    )
    cloth_rest = model.particle_q.numpy()
    rod_rest = np.asarray(rod.points, dtype=np.float32)
    rest = np.concatenate((cloth_rest, rod_rest))
    cloth_nodes = len(cloth_rest)
    triangles, cloth_structural = _cloth_topology(model)
    cloth_edges, cloth_kinds = _cloth_edges(model, cloth_structural)
    cable_edges, cable_kinds = _cable_topology(cable_segments)
    edges = np.concatenate((cloth_edges, cable_edges + cloth_nodes))
    kinds = np.concatenate((cloth_kinds, cable_kinds))
    # Capsule masses map conservatively to adjacent exported centerline nodes.
    segment_mass = model.body_mass.numpy()[bodies]
    rod_mass = np.pad(segment_mass * 0.5, (0, 1)) + np.pad(segment_mass * 0.5, (1, 0))
    inverse_mass = np.concatenate((model.particle_inv_mass.numpy(), 1 / rod_mass))
    acceleration = np.zeros((len(time), len(rest), 3), dtype=np.float32)
    wrench = np.zeros((len(time), model.body_count, 6), dtype=np.float32)
    cable_decode = _cable_decoder(rod_rest, bodies)

    def decode(state):
        cable = cable_decode(state)
        cable["position"] = np.concatenate((state.particle_q.numpy(), cable["position"]))
        cable["velocity"] = np.concatenate((state.particle_qd.numpy(), cable["velocity"]))
        return cable

    force_buffer = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
    records = _record(
        model,
        solver,
        dt,
        wrench,
        force_buffer,
        decode,
        force_field="body_f",
        substeps=_SUBSTEPS,
        collision_pipeline=pipeline,
    )
    # A freely moving cable has no imposed loads besides gravity. Its native
    # total COM momentum change therefore gives a contact-reaction diagnostic.
    # VBD iteration and integration residuals also enter this estimate.
    cable_momentum = (records["segment_center_velocity"].astype(np.float64) * segment_mass[None, :, None]).sum(axis=1)
    records["cable_reaction_impulse_estimate"] = (
        np.diff(cable_momentum, axis=0) - dt * float(segment_mass.sum(dtype=np.float64)) * np.asarray(_GRAVITY)
    ).astype(np.float32)
    result = _result(
        "mixed",
        seed,
        time,
        dt,
        records,
        rest,
        edges,
        kinds,
        inverse_mass,
        acceleration,
        {
            "solver": "SolverVBD",
            "scene": "unsupported rod dropped onto a sheet pinned at opposite edges",
            "control": "gravity only",
            "cloth_cells": [nx, ny],
            "cloth_length_m": length,
            "cloth_width_m": width,
            "cloth_mass_per_free_vertex_kg": cloth_mass,
            "cloth_membrane_stiffness": cloth_stiffness,
            "cloth_membrane_damping": 1.1,
            "cable_segments": cable_segments,
            "cable_radius_m": cable_radius,
            "cable_length_m": cable_length,
            "cable_density_kg_m3": cable_density,
            "cable_mass_kg": float(segment_mass.sum(dtype=np.float64)),
            "cable_stretch_stiffness_n_m": stretch,
            "cable_bend_stiffness_nm_rad": bend,
            "cable_release_gap_m": drop,
            "contact_enabled": enable_contact,
            "contact_observable": "maximum soft contact candidate count over the native substeps ending at this frame",
            "reaction_observable": "net cable COM momentum residual after gravity, world N·s per recorded interval",
        },
    )
    result["material_id"] = np.concatenate(
        (np.full(cloth_nodes, 1, dtype=np.int8), np.full(len(rod_rest), 2, dtype=np.int8))
    )
    result["contact_radius"] = np.concatenate(
        (np.full(cloth_nodes, cloth_radius, dtype=np.float32), np.full(len(rod_rest), cable_radius, dtype=np.float32))
    )
    result["triangles"] = triangles
    return result


def generate_episode(
    modality,
    seed,
    *,
    frames=121,
    dt=0.02,
    device="cpu",
    mixed_contact=True,
    late_pulse=False,
    additional_load=None,
    additional_load_at_middle=False,
):
    """Simulate one seeded deformable scene and return native state histories.

    Nodes are cloth vertices or cable centerline points. Fixed nodes have zero
    inverse mass. `external_acceleration[t]` describes the control applied from
    state t to t+1. Cable orientation and angular velocity are native segment
    diagnostics. A mixed scene may disable capsule-cloth contact for a paired
    trajectory check using the same seed and all other parameters. With
    `late_pulse`, isolated cloth or cable receives a second smooth load after
    the original schedule has ended.
    `additional_load` can specify one xyz acceleration per recorded frame
    to apply at the existing load location in an isolated scene.
    `additional_load_at_middle` moves only that added force to the cloth's
    middle row or the cable's middle segment, preserving the force vector.
    """
    if modality not in ("cloth", "cable", "mixed"):
        raise ValueError(f"Unsupported Newton teacher modality: {modality}")
    if frames < 2 or dt <= 0 or not math.isfinite(dt):
        raise ValueError("Use at least two frames and a finite positive interval")
    if late_pulse and (modality == "mixed" or (frames - 1) * dt < 3.65):
        raise ValueError("The late load pulse requires an isolated object and at least 3.65 s of data")
    if additional_load is not None:
        additional_load = np.asarray(additional_load, dtype=np.float32)
        if modality == "mixed" or additional_load.shape != (frames, 3) or not np.isfinite(additional_load).all():
            raise ValueError("The added load requires isolated cloth or cable and a finite [frames, 3] array")
    if additional_load_at_middle and additional_load is None:
        raise ValueError("Moving the added load requires an additional load")
    time = np.arange(frames, dtype=np.float64) * dt
    rng = np.random.default_rng(seed)
    generator = _cloth if modality == "cloth" else _cable
    with wp.ScopedDevice(device):
        if modality == "mixed":
            return _mixed(seed, rng, time, dt, device, enable_contact=mixed_contact)
        return generator(
            seed,
            rng,
            time,
            dt,
            device,
            late_pulse=late_pulse,
            additional_load=additional_load,
            additional_load_at_middle=additional_load_at_middle,
        )
