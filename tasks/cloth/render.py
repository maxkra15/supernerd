"""Render a fixed held-out fold from its measured native mesh positions."""

from pathlib import Path

import numpy as np
from folding import DT, FRAMES, NX, NY, RELEASE
from PIL import Image, ImageDraw, ImageFont


def font(n):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n)


def render(root):
    ex = np.load(root / "hybrid-example.npz")
    data = [ex["q"], ex["guess"]]
    length, width, *_ = ex["geometry"]
    W, H = 900, 340
    u = np.array([0.91, 0.41, 0])
    v = np.array([-0.17, 0.38, 0.91])
    depth = np.cross(u, v)
    geom = np.arange((NX + 1) * (NY + 1)).reshape(NY + 1, NX + 1)
    polys = np.stack(
        [geom[:-1, :-1], geom[:-1, 1:], geom[1:, 1:], geom[1:, :-1]], -1
    ).reshape(-1, 4)
    ink = "#183b42"

    def project(q, offset):
        centered = q - np.array([length / 2, 0, 0])
        return np.stack(
            (centered @ u * 455 + offset + W / 4, 206 - centered @ v * 455), -1
        )

    def tu(q):
        return [(float(x), float(y)) for x, y in q]

    frames = []
    for t in range(FRAMES + 1):
        image = Image.new("RGB", (W, H), "#f8faf7")
        draw = ImageDraw.Draw(image)
        for col, (arr, label) in enumerate(
            zip(data, ["Newton teacher", "Learned fold response"])
        ):
            offset = col * W / 2
            q = arr[t]
            draw.text((offset + 19, 14), label, fill=ink, font=font(20), stroke_width=0)
            lim = 0.08
            corners = np.array(
                [
                    [-lim, -width / 2 - lim, 0],
                    [length + lim, -width / 2 - lim, 0],
                    [length + lim, width / 2 + lim, 0],
                    [-lim, width / 2 + lim, 0],
                ]
            )
            draw.polygon(
                tu(project(corners, offset)), fill="#e4e9e5", outline="#afc0b9"
            )
            for xx in np.arange(0, length + 0.05, 0.06):
                draw.line(
                    tu(
                        project(
                            np.array(
                                [[xx, -width / 2 - lim, 0], [xx, width / 2 + lim, 0]]
                            ),
                            offset,
                        )
                    ),
                    fill="#cfdbd5",
                )
            for yy in np.arange(-width / 2, width / 2 + 0.02, 0.06):
                draw.line(
                    tu(
                        project(
                            np.array([[-lim, yy, 0], [length + lim, yy, 0]]), offset
                        )
                    ),
                    fill="#cfdbd5",
                )
            points = project(q, offset)
            light = np.array([0.2, -0.4, 0.89])
            pixels = np.asarray(image).copy()
            zbuffer = np.full((H, W), -np.inf)
            for i, idx in enumerate(polys):
                patch = q[idx]
                norm = np.cross(patch[1] - patch[0], patch[-1] - patch[0])
                norm = norm / max(np.linalg.norm(norm), 1e-9)
                tone = 0.57 + 0.43 * abs(norm @ light)
                base = (
                    np.array([225, 143, 76])
                    if i % NX >= NX // 2
                    else np.array([68, 156, 155])
                )
                rgb = base * tone
                for triangle, edges in (([0, 1, 2], (0, 2)), ([0, 2, 3], (0, 1))):
                    vertices = idx[triangle]
                    xy = points[vertices]
                    depths = q[vertices] @ depth
                    lo = np.maximum(np.floor(xy.min(0)).astype(int), [int(offset), 0])
                    hi = np.minimum(
                        np.ceil(xy.max(0)).astype(int), [int(offset) + W // 2, H]
                    )
                    if np.any(lo >= hi):
                        continue
                    yy, xx = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
                    direction1 = xy[1] - xy[0]
                    direction2 = xy[2] - xy[0]
                    det = direction1[0] * direction2[1] - direction1[1] * direction2[0]
                    if abs(det) < 1e-9:
                        continue
                    lx = xx + 0.5 - xy[0, 0]
                    ly = yy + 0.5 - xy[0, 1]
                    b1 = (lx * direction2[1] - ly * direction2[0]) / det
                    b2 = (direction1[0] * ly - direction1[1] * lx) / det
                    bary = np.stack([1 - b1 - b2, b1, b2])
                    depth_map = np.einsum("i,ijk->jk", depths, bary)
                    section = np.s_[lo[1] : hi[1], lo[0] : hi[0]]
                    zb = zbuffer[section]
                    inside = (bary.min(axis=0) >= -1e-8) & (depth_map > zb)
                    zb[inside] = depth_map[inside]
                    aa = [
                        abs(det)
                        / max(0.01, np.linalg.norm(xy[(j + 1) % 3] - xy[(j + 2) % 3]))
                        for j in edges
                    ]
                    edge = (
                        np.minimum(bary[edges[0]] * aa[0], bary[edges[1]] * aa[1])
                        < 0.45
                    )
                    region = pixels[section]
                    region[inside] = rgb.clip(0, 255).astype(int)
                    region[inside & edge] = (rgb * 0.83).clip(0, 255).astype(int)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            if t <= RELEASE:
                for idx in [0, NX, NY * (NX + 1), (NY + 1) * (NX + 1) - 1]:
                    x, y = points[idx]
                    draw.ellipse(
                        (x - 3, y - 3, x + 3, y + 3), fill="#243d44", outline="#f4f4f4"
                    )
            draw.line((offset + 20, 286, offset + W / 2 - 20, 286), fill="#d1d9d1")
        draw.line((W // 2, 10, W // 2, 280), fill="#c8d4ce")
        draw.text(
            (19, 298),
            f"t = {DT * t:.2f} s     {'Corner grasps held' if t < RELEASE else 'Grasps released'}",
            fill=ink,
            font=font(16),
        )
        draw.text(
            (550, 299), "Same open-loop action · new fabric", fill=ink, font=font(15)
        )
        frames.append(image)
    frames[-1].save(root / "cloth-fold.png")
    frames[0].save(
        root / "cloth-fold.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[45] * FRAMES + [700],
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    render(Path(__file__).resolve().parent)
