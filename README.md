<h2 align="center"> <a href="https://xiaobiaodu.github.io/mobile-gs-project/">Mobile-GS: Real-time Gaussian Splatting for Mobile Devices</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">

[![project](https://img.shields.io/badge/Webpage-blue)](https://xiaobiaodu.github.io/mobile-gs-project/)
[![arXiv](https://img.shields.io/badge/Arxiv-2603.11531-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.11531)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/xiaobiaodu/Mobile-GS/blob/main/LICENSE)





## 😮 Highlights

![teaser](./asset/teaser.png)
<h4 align="center"> <a href="https://github.com/xiaobiaodu/Mobile-GS/blob/main/LICENSE">You are allowed to use this repository in commercial usage for free.</a></h4>



## 🚩 **Updates**

Welcome to **watch** 👀 this repository for the latest updates.

✅ **[2026.3.13]** : Release [project page](https://xiaobiaodu.github.io/mobile-gs-project/).

✅ **[2026.3.13]** : Code Release. 






## Setup

For installation:
On current Blackwell/CUDA 13 systems, use `uv`, Python 3.12, and the latest PyTorch nightly `cu130` wheels.
```shell
git clone git@github.com:xiaobiaodu/Mobile-GS.git

cd Mobile-GS
uv venv --python 3.12
source .venv/bin/activate

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
TORCH_CUDA_ARCH_LIST=12.0 uv pip install --no-build-isolation -r requirements.txt
```
`--no-build-isolation` is required because the local CUDA extension submodules build against the already-installed PyTorch package.

For a single-command bring-up:
```shell
./scripts/setup_uv_py312_cu130.sh
```

To verify imports, CUDA extensions, and every visible GPU afterwards:
```shell
.venv/bin/python scripts/verify_env.py
```

#### Install [TMC (GPCC)](https://github.com/MPEGGroup/mpeg-pcc-tmc13), and add `tmc3` to your environment variable or manually specify its location in [the code](https://github.com/maincold2/OMG/blob/main/utils/gpcc_utils.py) (lines 243 and 258, this script is sourced from [HAC++](https://github.com/YihangChen-ee/HAC-plus)).
For RAPIDS package troubleshooting, please refer to the [CUML Installation Guide](https://docs.rapids.ai/install/).

### GPU selection

The training scripts currently run one process on one CUDA device. They now respect `LOCAL_RANK`, so you can choose a device with:
```shell
CUDA_VISIBLE_DEVICES=1 python train.py ...
```
or from a launcher that sets `LOCAL_RANK`.

We used [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Tanks & Temples, and Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

## Running

### Pre-training (Mini-Splatting)

```shell
#for outdoor scenes (e.g., Mip-NeRF 360 outdoor and T&T scenes)
python pretrain.py -s <path to COLMAP>  -m  <model path> --eval --imp_metric outdoor --sh_degree 3   --iterations 30000
#for indoor scenes (e.g., Mip-NeRF 360 indoor and DB scenes)
python pretrain.py -s <path to COLMAP>  -m  <model path> --eval --imp_metric indoor --sh_degree 3   --iterations 30000

```

### Fine-tune
```shell
python train.py -s  <path to COLMAP> -m <model path>  --eval --start_checkpoint  <model path>/chkpnt30000.pth 

# To improve rendering perofmrance, you can use multi-view training from MVGS. It may cause longer training time and memory.
python train.py -s  <path to COLMAP> -m <model path>  --eval --start_checkpoint  <model path>/chkpnt30000.pth   --mv  3
```

## WebGPU Viewer

This repo now includes a self-contained static browser viewer in `docs/` that is suitable for GitHub Pages. It exports a Mobile-GS point cloud into compact browser assets and renders sorted billboard splats with:
- WebGPU in Chromium-class browsers when available
- WebGPU in Firefox when enabled and hardware allows it
- WebGL2 fallback in WebKit/Safari-class browsers

Current scope:
- Uses real Mobile-GS outputs (`x/y/z`, SH DC color, opacity, isotropic billboard radius)
- Ships a trained sample scene at `docs/assets/scenes/lego-mini.{json,bin}`
- Is designed for real-time inspection in the browser, not CUDA-rasterizer parity

Example flow:

```shell
# 1. Train or pretrain a scene so you have a point_cloud.ply
python pretrain.py -s data/nerf_synthetic/lego_mini -m output/webgpu_lego_mini_pretrain --eval --imp_metric indoor --sh_degree 3 --iterations 1000 -r 8 --save_iterations 1000 --checkpoint_iterations 1000 --test_iterations 1000 --port 6010

# 2. Export the trained PLY into the Pages viewer assets
python scripts/export_web_scene.py \
  --ply output/webgpu_lego_mini_pretrain/point_cloud/iteration_1000/point_cloud.ply \
  --cameras output/webgpu_lego_mini_pretrain/cameras.json \
  --output-dir docs/assets/scenes \
  --slug lego-mini \
  --title "Mobile-GS Lego Mini"

# 3. Serve docs/ locally
python3 -m http.server 8008 -d docs

# 4. Open the viewer
# http://127.0.0.1:8008/
```

Browser verification:

```shell
npm install
npx playwright install chromium firefox webkit
npm run test:viewer
```

The verification script saves screenshots to `artifacts/browser-tests/` and currently exercises Chromium, Firefox, and WebKit locally. On Linux, Chromium and Firefox are launched inside Xvfb so the GPU-backed render paths can still be exercised from this machine.

For GitHub Pages:
- point Pages at the `docs/` directory
- or enable the checked-in GitHub Actions workflow at `.github/workflows/pages.yml`
- keep scene assets under `docs/assets/scenes/`
- the checked-in `docs/index.html` will load the bundled `lego-mini` scene automatically

## Evaluation
```shell
python render.py -s <path to COLMAP> -m <model path> --decode
python metrics.py -m <model path> 
```
#### --decode
Rendering with the compressed file (comp.xz), otherwise using the ply file. The results are the same regardless of this option.



## 👍 **Acknowledgement**
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [Mini-Splatting](https://github.com/fatPeter/mini-splatting)
* [OMG](https://github.com/maincold2/OMG)
* [MVGS](https://github.com/xiaobiaodu/MVGS)



## BibTeX
```
@misc{du2026mobile-gs,
      title={Mobile-GS: Real-time Gaussian Splatting for Mobile Devices}, 
      author={Xiaobiao Du and Yida Wang and Kun Zhan and Xin Yu},
      year={2026},
      eprint={2603.11531},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.11531}, 
}
```
