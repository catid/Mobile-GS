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
    parser.add_argument("--max-splats", type=int, default=20000)
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
    scales = np.stack(
        [np.asarray(ply[f"scale_{idx}"], dtype=np.float32) for idx in range(3)],
        axis=1,
    )
    radius = np.exp(scales).max(axis=1)

    # Cull very weak splats and keep the best contributors for browser delivery.
    importance = opacity * radius
    if args.max_splats < len(importance):
        keep = np.argpartition(importance, -args.max_splats)[-args.max_splats:]
        keep = keep[np.argsort(importance[keep])[::-1]]
    else:
        keep = np.argsort(importance)[::-1]

    xyz = xyz[keep]
    rgb = rgb[keep]
    opacity = opacity[keep]
    radius = radius[keep]

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    scene_radius = float(np.linalg.norm(bbox_max - bbox_min) * 0.5)
    q95_radius = float(np.quantile(radius, 0.95))
    point_scale = float(np.clip(scene_radius * 0.025 / max(q95_radius, 1e-5), 0.45, 1.8))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name = f"{args.slug}.bin"
    manifest_name = f"{args.slug}.json"

    interleaved = np.empty((len(xyz), 8), dtype=np.float32)
    interleaved[:, 0:3] = xyz
    interleaved[:, 3] = radius
    interleaved[:, 4:7] = rgb
    interleaved[:, 7] = opacity
    interleaved.tofile(output_dir / binary_name)

    cameras = load_cameras(args.cameras) if args.cameras else []
    camera_defaults = compute_camera_defaults(cameras, center, scene_radius)

    manifest = {
        "version": 1,
        "slug": args.slug,
        "title": args.title,
        "binary": binary_name,
        "splatCount": int(len(xyz)),
        "bounds": {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "center": center.tolist(),
            "radius": scene_radius,
        },
        "camera": camera_defaults,
        "render": {
            "pointScale": point_scale,
            "alphaScale": 1.0,
            "sortIntervalMs": 120,
            "backgroundTop": [0.05, 0.09, 0.17, 1.0],
            "backgroundBottom": [0.7, 0.82, 0.92, 1.0],
        },
    }
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2))

    print(f"exported {len(xyz)} splats to {output_dir / manifest_name}")
    print(f"binary size: {(output_dir / binary_name).stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"point scale: {point_scale:.3f}")


if __name__ == "__main__":
    main()
