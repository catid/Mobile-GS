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

from utils.sh_utils import C0


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
    rgb = np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)
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
        rgb     = rgb[keep]
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

    # Binary layout per splat: 16 × float32 = 64 bytes (vec4-aligned)
    #   [0-2]  xyz position
    #   [3]    opacity (sigmoid-activated)
    #   [4-7]  quaternion (qw, qx, qy, qz) normalised
    #   [8-10] scale (exp-activated sx, sy, sz)
    #   [11]   padding
    #   [12-14] rgb colour (DC-SH activated)
    #   [15]   padding
    interleaved = np.zeros((n, 16), dtype=np.float32)
    interleaved[:, 0:3]  = xyz
    interleaved[:, 3]    = opacity
    interleaved[:, 4:8]  = quat         # qw qx qy qz
    interleaved[:, 8:11] = scale
    interleaved[:, 11]   = 0.0
    interleaved[:, 12:15] = rgb
    interleaved[:, 15]   = 0.0
    interleaved.tofile(output_dir / binary_name)

    cameras = load_cameras(args.cameras) if args.cameras else []
    camera_defaults = compute_camera_defaults(cameras, center, scene_radius)

    manifest = {
        "version": 2,
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
            "backgroundTop": [0.05, 0.09, 0.17, 1.0],
            "backgroundBottom": [0.7, 0.82, 0.92, 1.0],
        },
    }
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2))

    binary_mb = (output_dir / binary_name).stat().st_size / (1024 * 1024)
    print(f"Exported {n:,} splats  →  {output_dir / manifest_name}")
    print(f"Binary size: {binary_mb:.2f} MiB")


if __name__ == "__main__":
    main()
