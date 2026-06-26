# Reference — `comfy.ops` & model management

Background on the two systems a native port leans on most: `comfy.ops` (the layer
wrappers) and the `ModelPatcher`/offload system (loading, casting, relocation). You
don't need to master these to do a port, but knowing why they exist explains the
conventions.

## `comfy.ops`

`comfy.ops` provides drop-in replacements for `torch.nn` layers
(`Linear`, `Conv2d`, `Conv3d`, `Embedding`, `LayerNorm`, `GroupNorm`, …) whose
`forward` is **hooked** so the framework can:

- **Cast** weights to the compute dtype on the fly (fp32 stored → bf16 compute).
- **Relocate** weights from CPU/offload device to the compute device just-in-time,
  then release them — this is what enables low-VRAM / partial loading.
- Apply **LoRA / model patches** without rewriting the module.

Two flavors you'll see:

```python
import comfy.ops
ops = comfy.ops.disable_weight_init   # used inside model __init__: skip default init
                                      # because weights will be loaded from a checkpoint
```

Using `disable_weight_init` avoids wasting time initializing weights you're about to
overwrite with the state dict.

**Implication for your port:** any layer with loaded weights must come from `ops.*`.
If you bypass it (raw `nn.Linear`), that layer is invisible to casting/offload and
will fail or run slow under non-trivial memory conditions.

### Reading params outside forward

The hooks only fire on `forward`. If you read `module.weight` directly (e.g. a VQ
codebook lookup, or copying weights at load time), the param has **not** been
relocated/cast — it's wherever offload left it. Options:

1. Restructure so the read happens inside a hooked forward (best).
2. Declare `disable_offload = True` so the weights stay fully resident (correct, used
   by Cube3D's VAE and the audio VAEs).
3. Manually `.to(device)` — **avoid**; it fights the system and is the thing managed
   paths exist to prevent.

## ModelPatcher & object patches

Models are wrapped in a `ModelPatcher` that tracks weights, patches, and device
placement. Two things matter for ports:

- **`model.clone()` + `add_object_patch(path, value)`** is how you override a
  parameter/submodule **without mutating the loaded weights**. It composes with
  offload (the patch is applied through the managed path) and is reversible. Cube3D's
  `CubeCodebookPatch` uses this to inject the projected VQ codebook into the GPT's
  `wte.weight`.
- `model.get_model_object(path)` fetches a submodule/param by dotted path
  (`"diffusion_model.transformer.wte.weight"`).

Never do `model.diffusion_model.some.weight = ...` directly — it bypasses patch
tracking and offload.

## The managed VAE path

`comfy.sd.VAE.decode(samples, vae_options={})` and `.encode(pixels)`:

1. Estimate memory (`memory_used_decode`) and load the VAE via model management.
2. Move/cast `samples` to the VAE's working dtype + device.
3. Call `first_stage_model.decode(samples, **vae_options)`.
4. Apply `process_output` and a trailing `movedim(1, -1)`, return to the output
   device/dtype.

So your `first_stage_model.decode` should assume inputs already arrive on the right
device/dtype, and (for non-image output) pre-invert the trailing `movedim` — see
[guides/03-vae.md](../guides/03-vae.md).

## Where to look in core

| Concern | File |
|---------|------|
| Layer wrappers | `comfy/ops.py` |
| Model patcher / object patches | `comfy/model_patcher.py` |
| Loading, offload, memory scheduling | `comfy/model_management.py` |
| Base model + `extra_conds` | `comfy/model_base.py` |
| VAE managed decode/encode | `comfy/sd.py` (`class VAE`) |
| Detection | `comfy/model_detection.py` |
| Registration | `comfy/supported_models.py`, `comfy/supported_models_base.py` |
| Latent formats | `comfy/latent_formats.py` |
| Samplers | `comfy/k_diffusion/sampling.py`, `comfy/samplers.py` |
