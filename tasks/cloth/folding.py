"""Newton benchmark for a supported pick, fold and release with ideal corner grasps."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

NX, NY = 12, 8
FRAMES = 80
RELEASE = 53
DT = 0.02
SUBSTEPS = 3
RADIUS = 0.003
HOLD_HEIGHT = 0.005


def geometry(seed):
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(0.45, 0.61),
            rng.uniform(0.29, 0.40),
            rng.uniform(350, 650),
            rng.uniform(0.00015, 0.00065),
            rng.uniform(0.38, 0.65),
        ],
        dtype=np.float32,
    )


def action(seed):
    rng = np.random.default_rng(seed + 911200)
    return np.array(
        [rng.uniform(-0.08, 0.44), rng.uniform(0.23, 0.63), rng.uniform(0.017, 0.075)],
        dtype=np.float32,
    )


def teacher(seed, parameters, *, substeps=SUBSTEPS, iterations=6):
    import newton
    import warp as wp
    from newton import ParticleFlags

    wp.init()
    L, W, stretch, bend, density = geometry(seed)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, -W / 2, HOLD_HEIGHT),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=NX,
        dim_y=NY,
        cell_x=L / NX,
        cell_y=W / NY,
        mass=float(L * W * density / ((NX + 1) * (NY + 1))),
        tri_ke=stretch,
        tri_ka=stretch,
        tri_kd=0.006,
        edge_ke=bend,
        edge_kd=0.0008,
        particle_radius=RADIUS,
    )
    builder.color(include_bending=True)
    model = builder.finalize(device="cpu")
    model.soft_contact_ke = 1800.0
    model.soft_contact_kd = 0.02
    model.soft_contact_mu = 0.65
    state, nextstate = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        particle_enable_self_contact=True,
        particle_self_contact_margin=0.009,
        particle_self_contact_gap=0.004,
        particle_vertex_contact_buffer_size=64,
        particle_edge_contact_buffer_size=128,
    )
    collision = newton.CollisionPipeline(model)
    contacts = collision.contacts()
    base = state.particle_q.numpy().copy()
    fixed = [0, NY * (NX + 1)]
    grasp = [NX, (NY + 1) * (NX + 1) - 1]
    dynamic_flags = model.particle_flags.numpy().copy()
    grip = dynamic_flags.copy()
    grip[fixed + grasp] &= ~int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(grip)
    qs = [base.copy()]
    min_z = base[:, 2].min()
    max_stretch = 1.0
    contacts_max = 0
    for f in range(FRAMES):
        if f == RELEASE:
            model.particle_flags.assign(dynamic_flags)
        for k in range(substeps):
            if f < RELEASE:
                u = np.clip((f + (k + 1) / substeps - 4) / (RELEASE - 9), 0, 1)
                smooth = u * u * (3 - 2 * u)
                delta = np.array(
                    [
                        (parameters[0] - 1) * L * smooth,
                        0,
                        L * parameters[1] * np.sin(np.pi * smooth)
                        + (parameters[2] - HOLD_HEIGHT) * smooth,
                    ]
                )
                for s in [state, nextstate]:
                    xyz = s.particle_q.numpy()
                    xyz[grasp] = base[grasp] + delta
                    xyz[fixed] = base[fixed]
                    s.particle_q.assign(xyz)
            state.clear_forces()
            collision.collide(state, contacts)
            solver.step(state, nextstate, control, contacts, DT / substeps)
            state, nextstate = nextstate, state
        xyz = state.particle_q.numpy().copy()
        if not np.isfinite(xyz).all():
            raise ValueError("Non-finite cloth")
        qs.append(xyz)
        contacts_max = max(contacts_max, int(contacts.soft_contact_count.numpy()[0]))
        min_z = min(min_z, float(xyz[:, 2].min()))
        cloth = xyz.reshape(NY + 1, NX + 1, 3)
        max_stretch = max(
            max_stretch,
            np.linalg.norm(np.diff(cloth, axis=1), axis=-1).max() / (L / NX),
            np.linalg.norm(np.diff(cloth, axis=0), axis=-1).max() / (W / NY),
        )
    return np.asarray(qs), {
        "min_z_m": float(min_z),
        "max_edge_stretch": float(max_stretch),
        "max_table_candidates": int(contacts_max),
    }


def target(rest, length, landing):
    goal = rest.copy()
    crease = 0.5 * (1 + landing) * length
    moving = rest[:, 0] > crease
    goal[moving, 0] = 2 * crease - rest[moving, 0]
    goal[:, 2] = RADIUS
    goal[moving, 2] += 0.004
    return goal


def metrics(q, g, landing):
    length = g[0]
    goal = target(q[0], length, landing)
    end = q[-6:].mean(axis=0)
    error = float(np.sqrt(np.square(end - goal).sum(axis=-1).mean()) / length)
    corners = [NX, (NY + 1) * (NX + 1) - 1]
    corner_error = float(
        np.linalg.norm(end[corners] - goal[corners], axis=1).max() / length
    )
    height = float(np.quantile(end[:, 2], 0.95))
    speed = float(np.linalg.norm((q[-1] - q[-6]) / (5 * DT), axis=1).mean())
    return {
        "shape_error_percent": 100 * error,
        "corner_error_percent": 100 * corner_error,
        "final_height_p95_m": height,
        "tail_speed_m_s": speed,
        "success": bool(
            error <= 0.05 and corner_error <= 0.10 and height <= 0.035 and speed <= 0.05
        ),
    }


def capture_row(item):
    name, seed, parameters, output, substeps, iterations = item
    start = time.perf_counter()
    q, extra = teacher(seed, parameters, substeps=substeps, iterations=iterations)
    np.savez_compressed(
        output / f"{name}.npz",
        q=q,
        g=geometry(seed),
        a=parameters,
        seed=seed,
        seconds=time.perf_counter() - start,
        **extra,
    )
    return name


def capture(jobs, output, workers, substeps=SUBSTEPS, iterations=6):
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tasks = [(n, s, a, output, substeps, iterations) for n, s, a in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(capture_row, tasks))
    return time.perf_counter() - start


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--cases", type=int, default=128)
    cli.add_argument("--seed", type=int, default=22000)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--pilot", action="store_true")
    args = cli.parse_args()
    jobs = [
        (f"episode-{i:04}", args.seed + i, action(args.seed + i))
        for i in range(args.cases)
    ]
    if args.pilot:
        jobs = [
            ("fold", 11, np.array([0.1, 0.45, 0.035])),
            ("unchanged", 11, np.array([1.0, 0.0, HOLD_HEIGHT])),
        ]
    elapsed = capture(jobs, args.output, args.workers)
    (args.output / "capture.json").write_text(
        json.dumps({"episodes": len(jobs), "seconds": elapsed, "workers": args.workers})
        + "\n"
    )
    for name, seed, a in jobs[:3]:
        with np.load(args.output / f"{name}.npz") as z:
            print(
                name,
                z["seconds"],
                z["min_z_m"],
                z["max_edge_stretch"],
                metrics(z["q"], z["g"], 0.1),
                flush=True,
            )


if __name__ == "__main__":
    main()
