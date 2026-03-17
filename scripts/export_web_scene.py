#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData
import torch

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


def load_opacity_phi_state(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    ordered_keys = [
        "main.0.weight",
        "main.0.bias",
        "main.2.weight",
        "main.2.bias",
        "main.4.weight",
        "main.4.bias",
        "phi_output.0.weight",
        "phi_output.0.bias",
        "opacity_output.0.weight",
        "opacity_output.0.bias",
    ]
    return {key: state[key].detach().cpu().numpy().astype(np.float32) for key in ordered_keys}


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
    parser.add_argument("--opacity-phi", type=Path,
                        help="Optional path to opacity_phi_nn.pt. Defaults to a sibling of the PLY.")
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

    rest_props = [p.name for p in ply.properties if p.name.startswith("f_rest_")]
    n_rest = len(rest_props)
    if n_rest % 3 != 0:
        raise ValueError(f"Unexpected SH layout: found {n_rest} rest coefficients")
    coeffs_per_channel = 1 + (n_rest // 3)
    sh_degree = int(round(math.sqrt(coeffs_per_channel) - 1))
    if (sh_degree + 1) ** 2 != coeffs_per_channel:
        raise ValueError(f"Cannot infer SH degree from {coeffs_per_channel} coefficients per channel")

    if n_rest > 0:
        rest = np.stack(
            [np.asarray(ply[f"f_rest_{i}"], dtype=np.float32) for i in range(n_rest)],
            axis=1,
        ).reshape(len(sh_dc), 3, coeffs_per_channel - 1)
    else:
        rest = np.zeros((len(sh_dc), 3, 0), dtype=np.float32)

    sh_coeffs = np.zeros((len(sh_dc), coeffs_per_channel, 3), dtype=np.float32)
    sh_coeffs[:, 0, :] = sh_dc
    if coeffs_per_channel > 1:
        sh_coeffs[:, 1:, 0] = rest[:, 0, :]
        sh_coeffs[:, 1:, 1] = rest[:, 1, :]
        sh_coeffs[:, 1:, 2] = rest[:, 2, :]
    sh_flat = sh_coeffs.reshape(len(sh_dc), coeffs_per_channel * 3)

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
        sh_flat = sh_flat[keep]
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

    opacity_phi_path = args.opacity_phi or args.ply.with_name("opacity_phi_nn.pt")
    opacity_phi = load_opacity_phi_state(opacity_phi_path)
    if opacity_phi is not None and sh_degree != 1:
        raise ValueError(
            f"Opacity/phi export currently expects SH1 assets, but {args.ply} has SH degree {sh_degree}"
        )

    geom = np.zeros((n, 11), dtype=np.float32)
    geom[:, 0:3]  = xyz
    geom[:, 3]    = opacity
    geom[:, 4:8]  = quat
    geom[:, 8:11] = scale

    if opacity_phi is None:
        binary_version = 5
        sh = np.concatenate(
            [
                sh_coeffs[:, :, 0],
                sh_coeffs[:, :, 1],
                sh_coeffs[:, :, 2],
            ],
            axis=1,
        ).astype(np.float16)
    else:
        binary_version = 6
        sh = sh_flat.astype(np.float16)

    with open(output_dir / binary_name, "wb") as f:
        geom.tofile(f)
        sh.tofile(f)

    opacity_phi_name = None
    if opacity_phi is not None:
        opacity_phi_name = f"{args.slug}-opacity-phi.bin"
        with open(output_dir / opacity_phi_name, "wb") as f:
            for key in [
                "main.0.weight",
                "main.0.bias",
                "main.2.weight",
                "main.2.bias",
                "main.4.weight",
                "main.4.bias",
                "phi_output.0.weight",
                "phi_output.0.bias",
                "opacity_output.0.weight",
                "opacity_output.0.bias",
            ]:
                opacity_phi[key].astype(np.float32).tofile(f)

    cameras = load_cameras(args.cameras) if args.cameras else []
    camera_defaults = compute_camera_defaults(cameras, center, scene_radius)

    manifest = {
        "version": binary_version,
        "slug": args.slug,
        "title": args.title,
        "binary": binary_name,
        "splatCount": n,
        "shDegree": sh_degree,
        "shCoefficientsPerChannel": coeffs_per_channel,
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
    if opacity_phi_name is not None:
        manifest["opacityPhi"] = {
            "binary": opacity_phi_name,
            "inputDim": int(opacity_phi["main.0.weight"].shape[1]),
            "hiddenDims": [
                int(opacity_phi["main.0.weight"].shape[0]),
                int(opacity_phi["main.2.weight"].shape[0]),
                int(opacity_phi["main.4.weight"].shape[0]),
            ],
            "format": "linear_relu_relu_linear_heads_v1",
            "shLayout": "coeff-major-rgb",
        }
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2))

    binary_mb = (output_dir / binary_name).stat().st_size / (1024 * 1024)
    print(f"Exported {n:,} splats  →  {output_dir / manifest_name}")
    print(f"Binary size: {binary_mb:.2f} MiB")


if __name__ == "__main__":
    main()
