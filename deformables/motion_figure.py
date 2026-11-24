"""Render native-mesh Newton trajectories and decoded predictions from saved numerical states."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from native_rollouts import decode


def render(root, prediction_file, out, split, seed, packed=None, native_path=None):
    with np.load(packed or root / f"data/{split}.npz") as f:
        data = dict(f)
    with np.load(native_path or root / f"data/native-{split}.npz") as f:
        native = dict(f)
    with np.load(prediction_file) as f:
        predictions = f["q_from_0"]
        assert np.array_equal(data["identity"], f["identity"])
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.9), layout="constrained")
    lines = []
    for row, family in enumerate(("cable", "cloth")):
        key = f"{family}-{seed}-descending-chirp-6"
        i = data["identity"].tolist().index(key)
        rest = native[key + "_rest"]
        free = native[key + "_free"]
        state = decode(predictions[i], data["static"][i], rest, free)
        traces = (native[key], np.concatenate([native[key][:1], state]))
        nx = int(data["static"][i,3]) + 1
        ny = len(rest) // nx
        index = np.arange(nx*ny).reshape(ny,nx)
        edges = list(index) + (list(index.T) if ny > 1 else [])
        def project(x):
            return np.stack([x[...,0] - .65*x[...,1],x[...,2] + .30*x[...,1]],axis=-1)
        all_frames = project(np.concatenate(traces, axis=1))
        low,high=all_frames.min(axis=(0,1))-.035,all_frames.max(axis=(0,1))+.035
        for col, (trace, label) in enumerate(zip(traces, ("Newton VBD", "SuperNeRD"), strict=True)):
            ax = axes[row,col]
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect="equal")
            ax.set_title(f"{family.capitalize()} · {label}",fontsize=10,loc="left")
            ax.axis("off")
            xy = project(trace)
            ax.scatter(*project(rest[~free]).T,s=10,color="#485e65",zorder=4)
            for edge in edges:
                line, = ax.plot([],[],color=("#007076","#b15730")[col],lw=2 if ny==1 else .9)
                lines.append((line,xy[:,edge]))
    clock=fig.suptitle(" ",fontsize=10)
    def update(frame):
        for line,xy in lines:
            line.set_data(xy[frame,:,0],xy[frame,:,1])
        clock.set_text(f"Prescribed descending chirp   ·   {frame*.02:.2f} s")
        return [clock,*(line for line,_ in lines)]
    update(175)
    fig.savefig(out.with_suffix(".png"),dpi=125)
    movie=FuncAnimation(fig,update,frames=range(0,301,5),interval=100,blit=False)
    movie.save(out,writer=PillowWriter(fps=10),dpi=100)
    plt.close(fig)


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "fresh"), default="fresh")
    parser.add_argument("--seed", type=int, default=41540)
    parser.add_argument("--predictions", type=Path, default=root / "results/paired-fresh-rollouts.npz")
    parser.add_argument("--output", type=Path, default=root.parent / "figures/newton-supernerd.gif")
    parser.add_argument("--packed", type=Path, help="Use an explicit packed trajectory bank.")
    parser.add_argument("--native", type=Path, help="Use the corresponding native-vertex bank.")
    args = parser.parse_args()
    render(root, args.predictions, args.output, args.split, args.seed, args.packed, args.native)
