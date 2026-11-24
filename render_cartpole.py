"""Render paired feedback policy replays from saved Newton joint coordinates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
METHODS = (
    ("pathwise", "Differentiable policy", "#187c65"),
    ("ppo", "PPO on learned dynamics", "#526987"),
    ("newton_ppo", "PPO on Newton", "#9c5639"),
)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def render(replays: Path, output: Path, seed: int, case: int) -> None:
    with np.load(replays) as source:
        state = [source[f"{method}_{seed}_state"][:, case] for method, _, _ in METHODS]
        action = [source[f"{method}_{seed}_action"][:, case] for method, _, _ in METHODS]
    title = font(25)
    label = font(17)
    detail = font(14)
    frames = []
    scale = 34.0
    for frame in range(0, len(state[0]), 2):
        image = Image.new("RGB", (1056, 276), "#f3f5f5")
        pen = ImageDraw.Draw(image)
        pen.text((24, 14), f"Cartpole feedback on Newton · held-out case {case}", font=title, fill="#1d2728")
        for idx, (_, name, color) in enumerate(METHODS):
            left = 16 + 344 * idx
            center = left + 168
            ground = 200
            pen.rounded_rectangle((left, 56, left + 336, 244), radius=9, fill="#ffffff", outline="#d9e0e0")
            pen.text((left + 15, 71), name, font=label, fill="#283336")
            x, theta, velocity, omega = state[idx][frame]
            cart_x = center + x * scale
            cart_y = ground - 13
            theta = math.atan2(math.sin(theta), math.cos(theta))
            top_x = cart_x + scale * math.sin(theta)
            top_y = cart_y - scale * math.cos(theta)
            pen.line((center - 4 * scale, ground, center + 4 * scale, ground), fill="#9ca6a6", width=2)
            pen.line((center, ground - 5, center, ground + 5), fill="#bdc6c6", width=2)
            pen.rounded_rectangle((cart_x - 9, cart_y - 3, cart_x + 9, cart_y + 6), radius=2, fill="#273337")
            pen.line((cart_x, cart_y, top_x, top_y), fill=color, width=4)
            pen.ellipse((top_x - 3, top_y - 3, top_x + 3, top_y + 3), fill=color)
            force = 1500.0 * action[idx][min(frame, len(action[idx]) - 1)]
            pen.text(
                (left + 15, 217),
                f"θ {theta:+.2f} rad   ·   x {x:+.2f} m   ·   u {force:+.0f} N",
                font=detail,
                fill="#364348",
            )
            if abs(x) > 4 or abs(velocity) > 10 or abs(omega) > 10:
                pen.text((left + 15, 104), "Outside safety envelope", font=detail, fill="#a84030")
        pen.text((24, 253), f"t = {frame / 60.0:.2f} s   ·   policy seed {seed}", font=detail, fill="#445257")
        frames.append(image)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=33, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args()
    render(args.replays, args.output, args.seed, args.case)


if __name__ == "__main__":
    main()
