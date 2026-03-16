# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Architecture Notes

### Algorithm Overview
Mobile-GS is **not standard Gaussian splatting**. It is a new algorithm designed to render faster on mobile hardware by replacing the expensive depth sort with a learned network. The MLP (`phi`) learns order-independent blending weights so that the color accumulation `C/w_fg` does not depend on sort order. This is the core algorithmic contribution.

### Mobile-GS rendering pipeline
- **`pretrain.py`** — Mini-Splatting 30k iter, produces high-quality SH3 PLY. No MLP.
- **`train.py`** — Fine-tune from pretrain checkpoint, 60k iter total.
  - iter 0–35k: trains Gaussians + `OpacityPhiNN` (outputs per-Gaussian `phi` and `opacity`)
  - iter 35k: `construct_net()` activates `mlp_cont`/`mlp_view`/`mlp_dc` (color MLP)
  - iter 35k–60k: trains everything; color MLP learns from Gaussians
  - iter 59k: `apply_svq()` quantizes scales/rotations/appearance into codebooks

### The `_ms` rasterizer vs `_msori`
The `_ms` variant replaces standard alpha compositing with an order-independent formula:
- Standard (`_msori`): `C += color * alpha * T` — order-dependent (T = transmittance)
- Mobile-GS (`_ms`):  `C += color * alpha * weight`, `w_fg += alpha * weight`,
  `out = C/w_fg * (1-T) + T * bg`
  where `weight = expf(max_scale/depth) + phi/depth² + phi²`

All three output terms are order-independent:
- `C/w_fg` is a weighted average — commutative
- `T = ∏(1 - αᵢ)` is a product — commutative
- No early-exit on T in the kernel, so all splats per tile are always processed

The `_ms` rasterizer now sorts only by **tile ID** (no depth component in the key).
The `_msori` rasterizer retains depth sort as it uses order-dependent standard blending.
Validated: PSNR unchanged at 36.17 dB after removing depth sort from `_ms`.

### `capture()` / `restore()` limitation
`capture()` saves basic Gaussian parameters (xyz, features, scaling, rotation, opacity,
optimizer state) but **does NOT save** the MLP weights (`mlp_cont`, `mlp_view`, `mlp_dc`,
`opacity_phi_nn`) or `_features_static`/`_features_view`. These are in `optimizer_net`
(separate optimizer). A checkpoint restored via `restore()` cannot be evaluated with
`render_imp` — it will error on `opacity_phi_nn is None`.

For export: use the PLY saved by `scene.save()`, which (after our fix) writes
`_features_static` as DC colors when `net_enabled=True`. The pretrain PLY (SH3) gives
better web-viewer PSNR (36.5 dB) than the finetune PLY (24.9 dB DC-only) because the
finetune MLP weights are not saved.

### Fine-tune quality summary (_ms rasterizer, lego white-bg)
- Pretrain + SH3 + standard compositing:    36.17 dB (trained for this formula)
- Pretrain + SH3 + _ms formula:            21.00 dB (not trained for _ms)
- Fine-tuned + SH1 + _ms formula:          21.23 dB (phi MLP not in PLY, SH1 only)
- Fine-tuned PLY has SH1 (9 f_rest fields) because `onedownSHdegree()` reduces degree
  at fine-tune start and `net_enabled=True` replaces higher SH with MLP.
- **finetune_ms.py pass 1** (30k from pretrain SH3):  29.05 dB — `output/lego_wb_ms/`
- **finetune_ms.py pass 2** (30k from pass 1, 60k total): 29.32 dB — `output/lego_wb_ms_60k/`
- **Web viewer uses pass-2 SH3 PLY** (29.32 dB, 210,707 splats)
- Fine-tuned `.pth` checkpoint at `output/lego_wb_finetune/` has the full trained phi MLP.
- `finetune_ms.py` bug fixed: opacity_reset_interval=3000 fired at iter 30000 (final),
  resetting opacities to near-zero → 10.8 dB all-white renders. Fixed by skipping reset
  after `densify_until` (default 15000).

### Web viewer
- Serves `docs/` via GitHub Pages (`catid.github.io/Mobile-GS`)
- Uses `docs/assets/scenes/lego-mini.{bin,json}` — currently _ms 60k fine-tune (29.32 dB)
- Renderer: WebGPU preferred, WebGL2 fallback; both implement full 3DGS covariance projection
- **Two-pass Mobile-GS _ms formula** (no sort needed):
  - Pass 1: additive accumulation into rgba16float FBO — `(color*alpha_w, alpha_w)` per pixel
    `alpha_w = opacity * exp(-power) * alphaScale * weight`, `weight = exp(min(max_scale/depth, 20))`
  - Pass 2: compose — `coverage = 1-exp(-w_fg)`, `output = mix(bg, C/w_fg, coverage)`
- Note: pretrain model was trained with standard compositing (`render_impori`), so _ms rendering
  may look slightly different from offline renders. The full fine-tuned model (trained with `_ms`)
  will match correctly. Binary format v4: geometry (12 f32/splat) + SH (48 f32/splat).
- WebGPU compose pass uses `textureLoad` (not `textureSample`) to avoid unfilterable-float restriction.
  Canvas context is acquired AFTER initialize() to preserve WebGL2 fallback if WebGPU init fails.

### SSH / push
No GitHub SSH key on the dev machine (`id_ed25519_kuang2` is for `kuang2.lan` only).
To push: `ssh-keygen -t ed25519 -f ~/.ssh/id_github && cat ~/.ssh/id_github.pub`
then add to github.com/settings/keys, update `~/.ssh/config`, run `git push`.

## Execution

- Add subtasks as needed toward the goal.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
