# 07 — Optimizations & performance

Native ports get most optimization "for free" by using ComfyUI's systems correctly.
This guide covers how to stay on the fast path and where the knobs are. The theme:
**don't fight model management; declare your needs and let the framework optimize.**

## 1. Offload & low-VRAM (use `comfy.ops`)

ComfyUI's streaming offload keeps weights on CPU/lower-VRAM and casts/relocates them
to the compute device on demand by **hooking the `forward` of `comfy.ops` layers**.
The single most important optimization decision is therefore: **build every weighted
layer from `comfy.ops`** (`ops.Linear`, `ops.Conv2d`, `ops.Embedding`, …). If you do,
your model automatically supports partial loading and low-VRAM execution.

What breaks it:

- Raw `torch.nn.Linear`/`Conv2d` with loaded weights → not hooked → not offloadable.
- Reading a parameter's `.weight` **outside** its hooked forward (e.g. a VQ codebook
  looked up via `F.embedding(q, self.codebook.weight)`). The cast hooks never fire,
  so under partial load the param is on the wrong device. If you must do this, set
  `disable_offload = True` on the VAE/model (a correctness flag), as Cube3D and the
  audio VAEs do — see [guides/03-vae.md](03-vae.md) §4. Prefer restructuring to go
  through a hooked forward when feasible.

## 2. Dtype strategy

- Set `supported_inference_dtypes` to what the forward can actually run in. Common:
  fp32 weights + bf16/fp16 autocast on the heavy forward.
- Don't blindly inherit upstream's (original research code's) dtype choices — they're
  often heavy-handed (fp32 everywhere). Default to ComfyUI's lighter conventions and
  only pin a heavier dtype for a specific op when the lighter one **noticeably degrades
  or changes the output**, verified with your parity harness. Cube3D restricts the VAE
  to `working_dtypes = [torch.float32]` because the VQ lookup and occupancy query are
  genuinely unstable in bf16 — a justified pin, documented as a deviation, not a reflex.
- Norm layers can upcast internally (`x.float()` → compute → `.type_as(x)`) without
  hurting offload.
- Match upstream's autocast **boundaries**, not just the dtype — running conditioning
  inside vs outside the bf16 region changes results (see [guides/02](02-text-encoder.md)).

## 3. Memory estimation (`memory_used_decode` / `memory_usage_factor`)

Model management schedules loads using your VRAM estimates. Wrong estimates cause
either OOMs or needless offloading (slow).

- `memory_usage_factor` on the `supported_models` entry scales the main model's
  estimated footprint.
- `memory_used_decode = lambda shape, dtype: …` on the VAE must be sized by the
  dimension that **actually drives** decode memory. Cube3D sizes by `shape[-1]`
  (`num_tokens`) after switching to the `(B, 1, num_tokens)` layout — sizing by the
  wrong axis under- or over-estimates badly.

## 4. Attention

- Use `torch.nn.functional.scaled_dot_product_attention` (SDPA). It picks the best
  available backend (FlashAttention, mem-efficient, math) automatically and is what
  the rest of ComfyUI relies on.
- Pass `is_causal=` / `attn_mask=` correctly; don't materialize a full mask when
  `is_causal` suffices (faster, less memory).
- For autoregressive models, implement a **KV cache** so each step is O(1) in context
  rather than re-attending the whole sequence (the Cube3D GPT port does this).

### Reuse the shared Flux RoPE

If your model uses rotary position embeddings, **reuse `comfy.ldm.flux.math`'s `rope`
/ `apply_rope`** rather than writing a bespoke RoPE. Most ComfyUI models already use
it, and at inference `apply_rope` dispatches to **comfy-kitchen's optimized kernel**
(`comfy.quant_ops.ck.apply_rope`) — a hand-rolled implementation gets none of that.

```python
# comfy/ldm/flux/math.py
def apply_rope(xq, xk, freqs_cis):
    if comfy.model_management.in_training:
        return _apply_rope(xq, xk, freqs_cis)
    else:
        return comfy.quant_ops.ck.apply_rope(xq, xk, freqs_cis)  # comfy-kitchen kernel
```

The shared `rope(pos, dim, theta)` builds `freqs_cis` as a real-valued rotation
(`[cos, -sin, sin, cos]`), which is what the optimized kernel expects. A common
anti-pattern (seen in the Cube3D GPT port) is to faithfully copy upstream's
**complex-number** RoPE (`torch.polar` + `view_as_complex`); it's correct but bypasses
the kernel. Prefer adapting to the Flux representation, and if you switch, **verify
parity** — the pairing convention (adjacent dims) and `theta` must match upstream so
token outputs don't drift.

## 5. Dependencies — don't add one just for your model

A new pip dependency for a single model is a maintenance and install-size cost for
*everyone*. The Cube3D port originally added `scikit-image` purely for marching
cubes; the maintainer rejected it and it was replaced with a **vendored,
dependency-free, vectorized pure-PyTorch** marching cubes in
`comfy/ldm/cube/marching_cubes.py` (validated identical to `skimage method="lorensen"`).

Guidance:

- Prefer a small **vendored, dependency-free** implementation over a heavy dep.
- If you must vendor an algorithm, validate it against a reference (face count,
  surface distance, numerical tolerance) and say so in the commit.
- Watch for subtle correctness details when reimplementing (e.g. the vendored
  marching-cubes table emitted the opposite **winding** from skimage, flipping mesh
  normals — caught only by checking signed mesh volume).

## 6. Reuse fast paths, skip slow re-implementations

- Reuse existing tiled decode/encode (`decode_tiled`, `encode_tiled`) for large
  spatial outputs instead of writing your own chunker — unless your modality needs
  bespoke chunking (Cube3D chunks occupancy **queries** via `chunk_size`).
- Reuse `repeat_to_batch_size` rather than hand-rolling batch broadcasting.
- Cache-friendly: keep per-call setup out of inner loops; precompute positional
  encodings / RoPE tables once.

## 7. Verify performance claims

If you assert an optimization helps, measure it (VRAM via `nvidia-smi` / torch memory
stats, wall-clock on a fixed seed). Note the GPU and torch/CUDA build — results vary
across builds, and some "differences vs upstream" are just the torch build (e.g.
bf16 argmax near-tie flips).

See [reference/ops-and-model-management.md](../reference/ops-and-model-management.md)
for the underlying mechanics.
