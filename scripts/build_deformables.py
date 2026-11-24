"""Build the forward-physics paper and figures from the numbered experimental receipts."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from urllib.parse import urljoin
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "deformables/results"
FAMILIES = ("cable", "cloth")
COLORS = ("#006966", "#aa4a31")


def read(path):
    return json.loads(path.read_text())


def score(name, start=0):
    return read(RESULTS / f"{name}.json")[str(start)]["summary"]


def curves():
    unpaired = read(RESULTS / "modal-long.json")["budgets"]["192"]["curve"]
    paired = read(RESULTS / "modal-paired.json")["fit_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, family, color in zip(axes, FAMILIES, COLORS, strict=True):
        for runs, symbol, label in ((unpaired, "o--", "One load"), (paired, "o-", "Paired loads")):
            ax.plot([r["fit_s"] for r in runs], [r["dev"]["0"][family] for r in runs],
                    symbol, color=color, alpha=1 if label.startswith("Paired") else .6, label=label)
        ax.set(title=family.capitalize(), xlabel="GPU fitting time (seconds)", yscale="log", xlim=(0, 31))
        ax.set_yticks([1, 3, 10, 30, 100], labels=["1", "3", "10", "30", "100"])
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Development shape error (% of length)")
    axes[1].legend(frameon=False)
    fig.savefig(ROOT / "figures/deformable-learning.png", dpi=180)
    plt.close(fig)


def force_figure(base, paired):
    fig, axes = plt.subplots(1, 2, figsize=(9,3.6), layout="constrained")
    for ax, family, color in zip(axes, FAMILIES, COLORS, strict=True):
        def group(rows):
            by_seed = {}
            for row in rows:
                if row['family'] == family:
                    by_seed.setdefault(row["seed"], []).append(row["response_error_pct"])
            return {s: np.mean(v) for s,v in by_seed.items()}
        a, b = group(base), group(paired)
        assert a.keys() == b.keys()
        for seed in sorted(a):
            ax.plot([0,1], [a[seed],b[seed]], "o-", color=color, alpha=.55, lw=1.4, ms=4)
        ax.axhline(100,lw=1.1,linestyle="--",color="#333333")
        ax.set(title=family.capitalize(), xlim=(-.25,1.25), ylim=(0,None), xticks=[0,1],
               xticklabels=["One load","Paired loads"])
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Error in added motion (% of Newton response)")
    fig.savefig(ROOT / "figures/force-transfer.png", dpi=180)
    plt.close(fig)


def print_paper(html):
    """Keep local images and resolve PDF links to their public web destinations."""
    public_url = "https://maxkra15.github.io/supernerd/"
    linked = re.sub(r'(<a href=")([^"]+)(")',
                    lambda m: m[1] + urljoin(public_url, m[2]) + m[3], html)
    with tempfile.TemporaryDirectory() as profile, tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", dir=ROOT
    ) as document:
        document.write(linked)
        document.flush()
        subprocess.run(
            ["google-chrome", "--headless", "--no-sandbox", "--disable-dev-shm-usage",
             "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
             f"--user-data-dir={profile}", f"--print-to-pdf={ROOT / 'paper/deformable-physics.pdf'}",
             Path(document.name).as_uri()], check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="Also print the manuscript with Google Chrome.")
    args = parser.parse_args()
    curves()
    a = read(RESULTS / "native-modal-long-fresh.json")["0"]["force_pairs"]
    b = read(RESULTS / "native-paired-fresh.json")["0"]["force_pairs"]
    force_figure(a,b)
    old = [
        ("NeurALL published load-diversity model", "Earlier fit", "native-neurall"),
        ("Streaming attention, short fit", "3.39 s", "native-stream"),
        ("Streaming attention, recursive fit", "47.82 s", "native-stream-long"),
        ("Damped modes, short fit", "3.26 s", "native-modal"),
        ("Damped modes, long fit", "25.03 s", "native-modal-long"),
        ("Damped modes, paired loads", "28.86 s", "native-paired"),
    ]
    fresh = [
        ("One load", "Release, 6 s", score("native-modal-long-fresh")),
        ("Paired loads", "Release, 6 s", score("native-paired-fresh")),
        ("Paired loads", "Moving restart, 4 s", score("native-paired-fresh", 100)),
    ]
    rows = [(name, clock, score(key)) for name,clock,key in old]
    asset = read(ROOT / "deformables/asset/results/fresh.json")["results"]
    env = Environment(loader=FileSystemLoader(ROOT / "paper"), autoescape=select_autoescape(default=True))
    gradient_results = ROOT / "deformables/gradients/results"
    retention = read(gradient_results / "retention/mean.json")
    clock = read(gradient_results / "fit-capture.json")["capture_including_compression_wall_s"]
    html = env.get_template("deformable-physics.html.j2").render(rows=rows, fresh=fresh, asset=asset,
                                                           retention=retention, gradient_clock=clock)
    (ROOT / "index.html").write_text(html)
    if args.pdf:
        print_paper(html)


if __name__ == "__main__":
    main()
