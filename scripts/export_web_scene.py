#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sh_utils import C0  # noqa: F401 (kept for potential fallback use)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def load_cameras(camera_path: Path) -> list[dict]:
    if not camera_path.exists():
        return []
    return json.loads(camera_path.read_text())


def compute_camera_defaults(cameras: list[dict], target: np.ndarray, scene_radius: float) -> dict:
    if cameras:
        first = cameras[0]
        eye = np.asarray(first["position"], dtype=np.float32)
        distance = float(np.linalg.norm(eye - target))
        if distance < 1e-4:
            distance = max(scene_radius * 2.4, 2.0)
        direction = (eye - target) / distance
        yaw = float(math.atan2(direction[0], direction[2]))
        pitch = float(math.asin(np.clip(direction[1], -1.0, 1.0)))
        fov_y = float(2.0 * math.atan(first["height"] / (2.0 * first["fy"])))
        return {
            "yaw": yaw,
            "pitch": pitch,
            "distance": distance,
            "fovY": fov_y,
        }

    return {
        "yaw": 0.9,
        "pitch": 0.45,
        "distance": max(scene_radius * 2.6, 2.4),
        "fovY": math.radians(46.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Mobile-GS PLY into a static browser scene asset.")
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--cameras", required=False, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max-splats", type=int, default=0,
                        help="Maximum splats to export (0 = keep all)")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white background (for models trained with white_background=True)")
    args = parser.parse_args()

    ply = PlyData.read(args.ply)["vertex"]

    xyz = np.stack(
        [
            np.asarray(ply["x"], dtype=np.float32),
            np.asarray(ply["y"], dtype=np.float32),
            np.asarray(ply["z"], dtype=np.float32),
        ],
        axis=1,
    )
    sh_dc = np.stack(
        [
            np.asarray(ply["f_dc_0"], dtype=np.float32),
            np.asarray(ply["f_dc_1"], dtype=np.float32),
            np.asarray(ply["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )

    # Load SH rest coefficients if available (45 = 15 per channel × 3 channels for SH3)
    rest_props = [p.name for p in ply.properties if p.name.startswith("f_rest_")]
    n_rest = len(rest_props)
    if n_rest >= 45:
        rest = np.stack(
            [np.asarray(ply[f"f_rest_{i}"], dtype=np.float32) for i in range(45)],
            axis=1,
        )  # N, 45: R_1..R_15, G_1..G_15, B_1..B_15
    else:
        rest = np.zeros((len(sh_dc), 45), dtype=np.float32)

    # SH per channel: [dc, rest_0..14] = 16 coefficients
    sh_r = np.concatenate([sh_dc[:, 0:1], rest[:, 0:15]], axis=1)   # N, 16
    sh_g = np.concatenate([sh_dc[:, 1:2], rest[:, 15:30]], axis=1)  # N, 16
    sh_b = np.concatenate([sh_dc[:, 2:3], rest[:, 30:45]], axis=1)  # N, 16

    opacity = sigmoid(np.asarray(ply["opacity"], dtype=np.float32))

    # Quaternion: PLY stores (rot_0=qw, rot_1=qx, rot_2=qy, rot_3=qz)
    quat_raw = np.stack(
        [np.asarray(ply[f"rot_{i}"], dtype=np.float32) for i in range(4)],
        axis=1,
    )
    # Normalize quaternions
    quat_norms = np.linalg.norm(quat_raw, axis=1, keepdims=True)
    quat = quat_raw / np.maximum(quat_norms, 1e-8)

    # Scale: PLY stores log-scale; exponentiate to get positive scales
    scale = np.exp(np.stack(
        [np.asarray(ply[f"scale_{i}"], dtype=np.float32) for i in range(3)],
        axis=1,
    ))

    # Optional culling by importance (opacity × max-scale)
    if args.max_splats > 0 and args.max_splats < len(xyz):
        importance = opacity * scale.max(axis=1)
        keep = np.argpartition(importance, -args.max_splats)[-args.max_splats:]
        keep = keep[np.argsort(importance[keep])[::-1]]
        xyz     = xyz[keep]
        sh_r    = sh_r[keep]
        sh_g    = sh_g[keep]
        sh_b    = sh_b[keep]
        opacity = opacity[keep]
        quat    = quat[keep]
        scale   = scale[keep]

    n = len(xyz)
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    scene_radius = float(np.linalg.norm(bbox_max - bbox_min) * 0.5)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name  = f"{args.slug}.bin"
    manifest_name = f"{args.slug}.json"

    # Binary format v5: two sections — geometry as float32, SH as float16
    # Section 1: geometry (11 floats per splat, float32)
    #   [0-2]  xyz position
    #   [3]    opacity (sigmoid-activated)
    #   [4-7]  quaternion (qw, qx, qy, qz) normalised
    #   [8-10] scale (exp-activated sx, sy, sz)
    #   (origIndex removed — renderer uses instance ID instead)
    # Section 2: SH (48 float16 per splat): sh_r[0..15], sh_g[0..15], sh_b[0..15]
    #   Float16 SH: max quantization error ~0.007, below 1-bit display precision
    geom = np.zeros((n, 11), dtype=np.float32)
    geom[:, 0:3]  = xyz
    geom[:, 3]    = opacity
    geom[:, 4:8]  = quat
    geom[:, 8:11] = scale

    # Section 2: SH (48 float16 per splat): sh_r[0..15], sh_g[0..15], sh_b[0..15]
    sh = np.concatenate([sh_r, sh_g, sh_b], axis=1).astype(np.float16)

    with open(output_dir / binary_name, "wb") as f:
        geom.tofile(f)
        sh.tofile(f)

    cameras = load_cameras(args.cameras) if args.cameras else []
    camera_defaults = compute_camera_defaults(cameras, center, scene_radius)

    manifest = {
        "version": 5,
        "slug": args.slug,
        "title": args.title,
        "binary": binary_name,
        "splatCount": n,
        "bounds": {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "center": center.tolist(),
            "radius": scene_radius,
        },
        "camera": camera_defaults,
        "render": {
            "alphaScale": 1.0,
            "sortIntervalMs": 120,
            "backgroundTop": [1.0, 1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0, 1.0],
            "backgroundBottom": [1.0, 1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0, 1.0],
        },
    }
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2))

    binary_mb = (output_dir / binary_name).stat().st_size / (1024 * 1024)
    print(f"Exported {n:,} splats  →  {output_dir / manifest_name}")
    print(f"Binary size: {binary_mb:.2f} MiB")


if __name__ == "__main__":
    main()
