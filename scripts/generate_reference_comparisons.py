#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INDICES = [0, 33, 66, 99, 132, 165]
HEADER_HEIGHT = 72
PADDING = 20
GUTTER = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Build docs-side comparison composites from evaluated test renders.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=55000)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/reference-comparisons"))
    parser.add_argument("--indices", nargs="+", type=int, default=DEFAULT_INDICES)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    args = parse_args()
    method = f"ours_{args.iteration}"
    base_dir = args.model_path / "test" / method
    render_dir = base_dir / "renders"
    gt_dir = base_dir / "gt"
    results = json.loads((args.model_path / "results.json").read_text())
    per_view = json.loads((args.model_path / "per_view.json").read_text())

    overall_psnr = float(results[method]["PSNR"])
    overall_ssim = float(results[method]["SSIM"])
    overall_lpips = float(results[method]["LPIPS"])
    psnr_per_view = per_view[method]["PSNR"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_label = load_font(28)
    font_title = load_font(30)

    items = []
    for index in args.indices:
        render_name = f"{index:05d}.png"
        image_name = f"r_{index}"
        render_path = render_dir / render_name
        gt_path = gt_dir / render_name
        if not render_path.exists() or not gt_path.exists():
            raise FileNotFoundError(f"Missing render pair for {render_name}")

        gt_image = Image.open(gt_path).convert("RGB")
        render_image = Image.open(render_path).convert("RGB")
        if gt_image.size != render_image.size:
            raise ValueError(f"Size mismatch for {render_name}: {gt_image.size} vs {render_image.size}")

        width, height = gt_image.size
        canvas_width = width * 2 + GUTTER + PADDING * 2
        canvas_height = height + HEADER_HEIGHT + PADDING * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), (10, 15, 25))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((0, 0, canvas_width - 1, canvas_height - 1), radius=28, fill=(10, 15, 25))
        draw.rectangle((0, 0, canvas_width, HEADER_HEIGHT + PADDING), fill=(14, 20, 34))

        left_x = PADDING
        right_x = PADDING + width + GUTTER
        image_y = PADDING + HEADER_HEIGHT
        canvas.paste(gt_image, (left_x, image_y))
        canvas.paste(render_image, (right_x, image_y))

        draw.text((left_x, 20), "Original", fill=(245, 247, 250), font=font_label)
        draw.text((right_x, 20), "Offline Render", fill=(245, 247, 250), font=font_label)

        psnr_value = float(psnr_per_view[render_name])
        metric_text = f"{image_name}  |  PSNR {psnr_value:.2f} dB"
        bbox = draw.textbbox((0, 0), metric_text, font=font_title)
        metric_width = bbox[2] - bbox[0]
        draw.text((canvas_width - PADDING - metric_width, 18), metric_text, fill=(255, 208, 126), font=font_title)

        separator_x = PADDING + width + GUTTER // 2
        draw.rounded_rectangle(
            (separator_x - 1, image_y, separator_x + 1, image_y + height),
            radius=2,
            fill=(53, 62, 84),
        )

        output_name = f"{image_name}.png"
        canvas.save(args.output_dir / output_name, quality=95)
        items.append({
            "index": index,
            "imageName": image_name,
            "file": output_name,
            "psnr": psnr_value,
            "renderFile": render_name,
        })

    mean_psnr = sum(item["psnr"] for item in items) / max(len(items), 1)
    summary = {
        "method": method,
        "overallMetrics": {
            "PSNR": overall_psnr,
            "SSIM": overall_ssim,
            "LPIPS": overall_lpips,
        },
        "representativeMeanPSNR": mean_psnr,
        "items": items,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
