"""
finetune_ms.py — Fine-tune a pretrain SH3 PLY with the _ms (order-independent)
rasterizer so the Gaussians learn the correct blending weights.

Unlike train.py, this script:
  - Loads from an existing PLY (GaussianModel, SH3, net_enabled=False)
  - Uses render_imp (_ms formula, phi=0 since no phi MLP)
  - Does NOT reduce SH degree or activate the color MLP
  - Does NOT densify (Gaussians already structured from pretrain)
  - Saves an SH3 PLY at the end that matches the web viewer formula

Usage:
  python finetune_ms.py \
    -s data/nerf_synthetic/lego \
    -m output/lego_wb_pretrain \
    --model_out output/lego_wb_ms \
    --white_background \
    --eval --quiet
"""

import os, sys, uuid
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from gaussian_renderer import render_imp as render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, load_iter, model_out,
             testing_iterations, saving_iterations, debug_from,
             densify_from, densify_until, densify_interval, densify_grad_threshold):

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(model_out, exist_ok=True)
    with open(os.path.join(model_out, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(dataset))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_out)

    # ── Load pretrain Gaussians (SH3, net_enabled=False) ──────────────────────
    gaussians = GaussianModel(dataset.sh_degree)
    # Load via Scene — reads PLY from dataset.model_path at load_iter
    scene = Scene(dataset, gaussians, load_iteration=load_iter,
                  shuffle=True, decode=False)
    print(f"Loaded {len(gaussians.get_xyz):,} Gaussians from iter {scene.loaded_iter}")
    assert not gaussians.net_enabled, "Expected net_enabled=False (SH3 PLY path)"

    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros(len(gaussians.get_xyz), device="cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Fine-tune (_ms)")

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_pt, visibility, radii = (render_pkg["render"],
            render_pkg["viewspace_points"], render_pkg["visibility_filter"],
            render_pkg["radii"])

        gt = viewpoint_cam.original_image.cuda()
        Ll1  = l1_loss(image, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}",
                                          "N": f"{len(gaussians.get_xyz):,}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar("train/l1_loss",    Ll1.item(),  iteration)
                tb_writer.add_scalar("train/total_loss", loss.item(), iteration)

            # ── Densification ─────────────────────────────────────────────────
            if densify_until > 0:
                gaussians.max_radii2D[visibility] = torch.max(
                    gaussians.max_radii2D[visibility], radii[visibility])
                gaussians.add_densification_stats(viewspace_pt, visibility)

                if (iteration > densify_from and iteration < densify_until
                        and iteration % densify_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0
                        and iteration < densify_until):
                    gaussians.reset_opacity()

            # ── Evaluate ──────────────────────────────────────────────────────
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                for split, cameras in [("test",  scene.getTestCameras()),
                                        ("train", scene.getTrainCameras()[:5])]:
                    if not cameras:
                        continue
                    psnr_acc = 0.0
                    for cam in cameras:
                        img = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0, 1)
                        psnr_acc += psnr(img, cam.original_image.cuda()).mean().item()
                    print(f"\n[{iteration}] {split} PSNR: {psnr_acc/len(cameras):.4f} dB")
                torch.cuda.empty_cache()

            # ── Save PLY ──────────────────────────────────────────────────────
            if iteration in saving_iterations:
                ply_path = os.path.join(model_out, "point_cloud",
                                        f"iteration_{iteration}", "point_cloud.ply")
                print(f"\n[{iteration}] Saving SH3 PLY → {ply_path}")
                gaussians.save_ply(ply_path)

            # ── Optimizer step ────────────────────────────────────────────────
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    # Final save
    ply_path = os.path.join(model_out, "point_cloud",
                            f"iteration_{opt.iterations}", "point_cloud.ply")
    print(f"\nFinal save → {ply_path}")
    gaussians.save_ply(ply_path)


if __name__ == "__main__":
    parser = ArgumentParser("finetune_ms parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iter",   type=int, default=-1,
                        help="Iteration of pretrain PLY to load (default: latest)")
    parser.add_argument("--model_out",   type=str, required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--debug_from",  type=int, default=-1)
    parser.add_argument("--test_iterations",  nargs="+", type=int,
                        default=[7000, 15000, 30000])
    parser.add_argument("--save_iterations",  nargs="+", type=int,
                        default=[30000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    print("Output:", args.model_out)
    safe_state(args.quiet)

    opt = op.extract(args)
    training(
        lp.extract(args), opt, pp.extract(args),
        args.load_iter, args.model_out,
        args.test_iterations, args.save_iterations, args.debug_from,
        opt.densify_from_iter, opt.densify_until_iter,
        opt.densification_interval, opt.densify_grad_threshold,
    )
    print("Done.")
