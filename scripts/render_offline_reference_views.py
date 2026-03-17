#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render_imp
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix


class OfflineCamera:
    def __init__(self, width, height, fov_y, world_view_transform, projection_matrix):
        self.image_width = width
        self.image_height = height
        self.FoVy = fov_y
        self.FoVx = 2.0 * math.atan(math.tan(fov_y * 0.5) * (width / height))
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def parse_args():
    parser = argparse.ArgumentParser(description="Render six offline reference views for the docs site.")
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path("output/lego_wb_ms_60k/point_cloud/iteration_30000/point_cloud.ply"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/assets/scenes/lego-mini.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/offline-views"),
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--opacity-phi", type=Path)
    return parser.parse_args()


def normalize(vector):
    length = np.linalg.norm(vector)
    if length == 0.0:
        return vector
    return vector / length


def look_at_matrix(eye, target, up):
    forward = normalize(target - eye)
    right = normalize(np.cross(up, forward))
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = normalize(np.cross(up, forward))
    true_up = normalize(np.cross(forward, right))
    z_axis = forward

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = z_axis
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = -np.dot(z_axis, eye)
    return view


def make_camera(center, eye, up, width, height, fov_y):
    view = torch.tensor(look_at_matrix(eye, center, up), dtype=torch.float32, device="cuda").transpose(0, 1)
    projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=2.0 * math.atan(math.tan(fov_y * 0.5) * (width / height)), fovY=fov_y)
    projection = projection.transpose(0, 1).to(device="cuda", dtype=torch.float32)
    return OfflineCamera(width, height, fov_y, view, projection)


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    center = np.array(manifest["bounds"]["center"], dtype=np.float32)
    distance = float(manifest["camera"]["distance"]) * args.distance_scale
    fov_y = float(manifest["camera"]["fovY"])
    bg = torch.tensor(manifest["render"]["backgroundTop"][:3], dtype=torch.float32, device="cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sh_degree = int(manifest.get("shDegree", args.sh_degree))
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(args.ply))
    opacity_phi_path = args.opacity_phi or args.ply.with_name("opacity_phi_nn.pt")
    if opacity_phi_path.exists():
        gaussians.init_vnn()
        gaussians.opacity_phi_nn.load_state_dict(torch.load(opacity_phi_path, map_location="cpu", weights_only=True))
        gaussians.opacity_phi_nn = gaussians.opacity_phi_nn.eval().cuda()

    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )

    views = [
        ("front", np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("back", np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("right", np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("left", np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("top", np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)),
        ("bottom", np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]

    with torch.no_grad():
        for name, direction, up in views:
            eye = center + direction * distance
            camera = make_camera(center, eye, up, args.width, args.height, fov_y)
            rendering = render_imp(camera, gaussians, pipe, bg)["render"].clamp(0.0, 1.0)
            torchvision.utils.save_image(rendering, args.output_dir / f"{name}.png")
            print(args.output_dir / f"{name}.png")


if __name__ == "__main__":
    main()
