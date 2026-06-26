# Case study — Roblox Cube3D (text → 3D)

The fully worked example behind most of this guide. Branch:
`feat/cube3d-native` in Comfy-Org/ComfyUI. Design thread:
[T-019ec361](https://ampcode.com/threads/T-019ec361-addb-70d8-a74b-438ce8a1e096).

## What Cube3D is

[Roblox/cube](https://github.com/Roblox/cube) (`cube3d-v0.5`) is **not** a diffusion
model. It's two cooperating networks:

- **`shape_gpt`** — a `DualStreamRoformer`, an **autoregressive** transformer that
  samples 1024 discrete VQ token IDs from CLIP-L text embeddings (custom
  linearly-decaying CFG + optional top-p).
- **`shape_tokenizer`** — a `OneDAutoEncoder`, a **decode-only** VQ-VAE that turns
  those IDs into an occupancy-logit grid → mesh via marching cubes.

They're coupled by a critical load-time step: the VQ codebook is projected through
`shape_proj` and injected into the GPT's `wte` embedding table, or generation is
garbage.

## The native graph

```
CLIPLoader(clip-l) → CLIPTextEncode ×2 → CFGGuider
UNETLoader(shape_gpt) → MODEL ── CubeCodebookPatch ← VAELoader(shape_tokenizer)
SamplerCube + BasicScheduler(steps=1) + RandomNoise + EmptyCubeLatent
   → SamplerCustomAdvanced → LATENT (token IDs)
VAEDecodeCube(VAE, LATENT) → MESH → SaveGLB
```

Modeled on the existing **Causal-WAN AR-video** precedent: the GPT loads as a normal
`MODEL` and generation runs through a dedicated `cube` sampler instead of KSampler.
Only four new nodes; everything else is reused.

## What changed in core

| File | Change |
|------|--------|
| `comfy/ldm/cube/gpt.py` | `DualStreamRoformer` port (dual-stream RoPE attention, per-head RMSNorm, SwiGLU, KV cache; `rope_theta=10000`). Uses `comfy.ops`. |
| `comfy/ldm/cube/vae.py` | `OneDAutoEncoder` **decode path** (codebook lookup → decoder → occupancy decoder → dense grid). |
| `comfy/ldm/cube/marching_cubes.py` | **vendored, dependency-free** vectorized Lorensen marching cubes. |
| `comfy/model_detection.py` | detect `shape_gpt` → `image_model="cube3d"`, dims from state dict. |
| `comfy/supported_models.py` | `Cube3D(BASE)`: `latent_format`, `clip_target()→None`, dtypes. |
| `comfy/model_base.py` | `Cube3D(BaseModel)`: `_apply_model` raises; `extra_conds`→`c_crossattn`. |
| `comfy/latent_formats.py` | `Cube3D` latent format (`latent_channels=1, latent_dimensions=1`). |
| `comfy/sd.py` | detect `shape_tokenizer` → build `CubeShapeVAE`; identity `process_output`; fp32; `disable_offload`. |
| `comfy/k_diffusion/sampling.py` | `sample_cube` autoregressive sampler. |
| `comfy_extras/nodes_cube.py` | `EmptyCubeLatent`, `CubeCodebookPatch`, `SamplerCube`, `VAEDecodeCube`. |

## The traps it hit (and the lessons)

Each maps to a guide section:

1. **Latent layout truncation.** A dummy `(B, num_tokens, 1)` latent got truncated by
   `fix_empty_latent_channels` (1024 → 4). Fixed by channels-first `(B, 1, num_tokens)`
   with `latent_channels=1` — zero core sampler changes. [→04](../guides/04-latent-format.md)
2. **Fighting model management.** `VAEDecodeCube` first did manual `load_models_gpu` +
   `.to(device)` and forced `disable_offload`. Refactored to route decode through
   `comfy.sd.VAE.decode` → `CubeShapeVAE.decode`. [→03](../guides/03-vae.md)
3. **`movedim` orientation.** `VAE.decode` applies a trailing `movedim(1,-1)`; the
   cube decode pre-inverts it so the node gets `(B, gx, gy, gz)` grid logits unchanged.
4. **Default clamp destroying the isosurface.** `process_output` set to identity (the
   default `[0,1]` clamp would wreck the signed occupancy logits marching cubes
   thresholds at `level=0.0`). [→03](../guides/03-vae.md)
5. **`disable_offload` for the right reason.** The VQ bottleneck reads raw params
   outside any hooked forward, so streaming offload can't relocate them → device
   mismatch. Kept `disable_offload=True` as a correctness flag (like audio VAEs), not
   a shortcut. [→07](../guides/07-optimizations.md)
6. **No pip dependency for one model.** `scikit-image` (added only for marching cubes)
   was rejected and replaced by a vendored pure-PyTorch implementation, validated
   identical to `skimage method="lorensen"`. [→07](../guides/07-optimizations.md)
7. **Mesh winding.** The vendored Lorensen table emits the opposite base winding from
   skimage; the upstream-style `faces[:, [2,1,0]]` flip produced inward normals
   (negative volume). Dropped the flip. (Caught by checking signed mesh volume.)
8. **Conditioning dtype.** `sample_cube` computes conditioning in fp32 outside the
   bf16 autocast to match upstream `Engine.prepare_inputs`; doing it inside bf16 gave a
   different mesh. [→02](../guides/02-text-encoder.md)
9. **Constant not in the state dict.** `rope_theta=10000` is upstream's released
   constant (the dataclass default is 1000) — hardcoded with a comment.
10. **Document the deviations.** The final commit added comments for every divergence
    (fp32 `working_dtypes`, `clip_target()→None`, `rope_theta`) and removed a dead
    flag. [→conventions](../reference/conventions.md)

## Parity results

Loading upstream and native `DualStreamRoformer` in the same process on a 2×4090 box:

- per-step GPT logits: **max-abs-diff = 0.0**
- VQ-VAE decode latents: matched to **3.7e-4**
- full graph produced a valid `pagoda.glb` (~109k–215k verts), deterministic on re-run.

When fed identical conditioning under the same torch build, output is **bit-identical**.

## Known out-of-the-box differences vs upstream (documented, not bugs)

- **torch/CUDA build** → bf16 greedy-argmax near-ties flip (~16%) → different shape.
- **ComfyUI SD1 CLIP padding/mask** differs from HF; padding positions diverge >1.0,
  and cube conditions on all 77 positions. The one remaining parity gap to close
  (replicate HF padding + padding attention mask). [→02](../guides/02-text-encoder.md)
- **Marching-cubes backend** (vendored classic Lorensen vs upstream warp Lorensen) —
  same algorithm family; minor diff at ambiguous cells.
- **No pymeshlab decimation** → ~10× more faces than upstream's default; compare with
  upstream `--disable-postprocessing`.

## Open items

- Close the CLIP padding/mask parity gap.
- (Optional) a `MeshDecimate` node to match upstream's default post-processing.
- Consolidation of the per-model 3D VAE decode nodes — see
  [investigations/3d-vae-node-consolidation.md](../investigations/3d-vae-node-consolidation.md).
