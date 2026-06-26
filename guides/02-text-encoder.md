# 02 — Text encoders & conditioning

Two questions decide this section:

1. Does the model ship its **own** text encoder, or can it reuse an existing one?
2. How does the conditioning tensor get from the encoder into your model's forward?

## Reuse before you port

Most models condition on a CLIP/T5/Llama family encoder ComfyUI already supports. If
so, **reuse it** — don't port a duplicate.

- Return `None` from `supported_models.<Model>.clip_target()`.
- Wire a standard `CLIPLoader` (+ the right `type=`) and `CLIPTextEncode` in the
  graph/blueprint.

Cube3D does exactly this: its checkpoint is GPT-only, so `clip_target()` returns
`None` and the graph uses `CLIPLoader(clip-l)` → `CLIPTextEncode`, emitting
`(B, 77, 768)` CLIP-L embeddings.

```python
def clip_target(self, state_dict={}):
    # No bundled text encoder; reuse a standard CLIPLoader/CLIPTextEncode.
    return None
```

## Porting a bundled text encoder

If the checkpoint bundles a novel encoder, port it under `comfy/text_encoders/` and
return a `ClipTarget`:

```python
def clip_target(self, state_dict={}):
    return supported_models_base.ClipTarget(
        comfy.text_encoders.<model>.<Tokenizer>,
        comfy.text_encoders.<model>.<ClipModel>,
    )
```

- Build encoder layers from `comfy.ops` like any other model.
- Detect it in `comfy/sd.py`'s CLIP detection the same way you detect the main model
  (unique keys, inferred dims).
- Reuse the existing tokenizer base classes where the vocab/merges match.

## How conditioning reaches the forward

The flow is:

```
CLIPTextEncode → CONDITIONING (a tensor + a dict of extras)
   → guider/sampler builds kwargs (cross_attn=..., pooled_output=..., etc.)
   → BaseModel.extra_conds(**kwargs) wraps them in comfy.conds.*
   → your network's forward receives them
```

Your `extra_conds` decides the keys. The common ones:

| Key | Wrapper | Typical use |
|-----|---------|-------------|
| `c_crossattn` | `CONDRegular` | cross-attention text embeddings |
| `y` | `CONDRegular` | pooled / class embedding |
| `guidance` | `CONDRegular` | distilled guidance scalar (Flux-style) |

```python
def extra_conds(self, **kwargs):
    out = {}
    cross_attn = kwargs.get("cross_attn", None)
    if cross_attn is not None:
        out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
    return out
```

For non-diffusion models that bypass `apply_model`, your **custom sampler** may read
the conditioning directly off the guider instead (see
[guides/05-sampler.md](05-sampler.md) — `sample_cube` reads `guider.conds`).

## Traps that cause parity drift

These are the conditioning bugs that actually bit real ports:

- **Wrong dtype / autocast region.** Upstream may compute conditioning in fp32
  *outside* the model's bf16 autocast. If you compute it inside bf16 you get a
  subtly different (and less correct) result. Cube3D's `sample_cube` does
  `encode_text` + `bbox_proj` in fp32 outside the bf16 autocast to match upstream
  `Engine.prepare_inputs`; an earlier version that didn't produced a different mesh.
- **Padding & attention masks.** ComfyUI's SD1 CLIP path pads and masks differently
  from HuggingFace. Real prompt tokens match to ~1e-3, but **padding positions can
  diverge by >1.0**. If your model conditions on *all* positions (including padding)
  rather than just the real tokens, that changes the output. The fix is to replicate
  HF padding + a padding attention mask. (This is the remaining open parity gap in
  the Cube3D port — see [case-studies/cube3d.md](../case-studies/cube3d.md).)
- **Pooled vs sequence output.** Make sure you're feeding the encoder output the
  model actually expects (last hidden state vs pooled vs a specific layer).

Next: [guides/03-vae.md](03-vae.md).
