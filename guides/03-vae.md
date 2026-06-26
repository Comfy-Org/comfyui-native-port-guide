# 03 — VAE / autoencoder

The VAE turns latents into the final modality (image, audio, voxel, mesh, splat) and
back. Native VAEs are **detected and constructed in `comfy/sd.py`** and run through
the **managed** `comfy.sd.VAE.decode`/`encode` path. The biggest, most common mistake
is bypassing that path.

## 1. Detection + construction (`comfy/sd.py`)

Add a branch in `VAE.__init__` keyed on a unique VAE state-dict key, inferring dims
from shapes. Real example (Cube3D shape tokenizer, decode-only):

```python
# Roblox Cube3D shape tokenizer (OneDAutoEncoder, decode-only)
elif "bottleneck.block.codebook.weight" in sd:
    self.latent_dim = 1
    # The VQ bottleneck (get_codebook/lookup_codebook) reads raw parameters
    # outside any hooked forward, so the streaming-offload cast hooks can't
    # relocate them; the model must be fully resident to decode. (see §4)
    self.disable_offload = True
    embed_dim = sd["bottleneck.block.codebook.weight"].shape[1]
    num_codes = sd["bottleneck.block.codebook.weight"].shape[0]
    width = sd["bottleneck.block.c_out.weight"].shape[0]
    # … infer the rest from shapes …
    self.first_stage_model = comfy.ldm.cube.vae.CubeShapeVAE(
        embed_dim=embed_dim, width=width, num_codes=num_codes, …)
    # Grid logits are float32 regardless of weight dtype; keep process_output
    # identity (the default clamps to [0,1] in-place and would destroy the isosurface).
    self.process_output = lambda image: image
    self.process_input = lambda image: image
    self.memory_used_decode = lambda shape, dtype: (1000 * shape[-1] * 768) * model_management.dtype_size(dtype)
    # fp32-only (unlike most VAEs that allow fp16/bf16): the VQ codebook lookup and
    # occupancy-grid query must run in fp32 to match upstream and stay stable.
    self.working_dtypes = [torch.float32]
```

The knobs you set here:

| Field | What it does | When to change it |
|-------|--------------|-------------------|
| `first_stage_model` | the ported AE module | always |
| `working_dtypes` | dtypes decode/encode may run in | restrict to `[fp32]` if codebook lookup / quantization needs it |
| `process_output` | post-decode transform | set to **identity** if the default `[0,1]` clamp would corrupt non-image output |
| `process_input` | pre-encode transform | identity for non-image inputs |
| `memory_used_decode` | VRAM estimate for scheduling | size by the dimension that actually drives memory |
| `disable_offload` | force full residency | only when weights are read outside a hooked forward (§4) |
| `latent_dim` | dimensionality hint | match your latent layout |

## 2. Route decode through the managed path

`comfy.sd.VAE.decode(samples, vae_options={...})` handles model loading, device
placement, and dtype casting, then calls `first_stage_model.decode(samples, **vae_options)`.
**Make `first_stage_model.decode` the entry point** and let the managed path do the
rest. Do **not** call `load_models_gpu(...)` + `.to(device)` by hand inside the node.

The Cube3D port originally did manual loading inside `VAEDecodeCube` and had to force
`disable_offload`; the fix was to move everything behind `CubeShapeVAE.decode` so the
managed path took over (commit "route VAE decode through managed comfy.sd.VAE.decode").

### The trailing `movedim` gotcha

`comfy.sd.VAE.decode` applies a trailing `movedim(1, -1)` to its output (it assumes
channels-first → channels-last like an image). If your decode returns something that
isn't an image (e.g. a `(B, gx, gy, gz)` occupancy grid), **pre-invert** that movedim
inside your `decode` so callers get the orientation they expect:

```python
def decode(self, samples, resolution_base=8.0, chunk_size=100_000, **kwargs):
    ids = samples.reshape(samples.shape[0], -1)[:, :self.cfg_num_encoder_latents]
    latents = self.decode_indices(ids.round().long().clamp(0, self.cfg_num_codes - 1))
    grid_logits, *_ = self.extract_geometry(latents, resolution_base=resolution_base, chunk_size=chunk_size)
    return grid_logits.movedim(-1, 1)   # pre-invert VAE.decode's trailing movedim(1, -1)
```

## 3. `process_output` — don't let the default clamp corrupt your data

The default `process_output` clamps to `[0, 1]` in place (correct for images). If
your decode returns **signed logits** (e.g. occupancy values that marching cubes
thresholds at `level=0.0`), that clamp **destroys the isosurface**. Set it to
identity, with a comment. Same goes for `process_input` on non-image encode inputs.

## 4. `disable_offload` — a correctness flag, not a shortcut

Streaming offload works by hooking `forward` to cast/relocate weights on the fly. If
your decode reads **raw parameters outside any hooked forward** — e.g. a VQ codebook
via `F.embedding(q, self.get_codebook())` — those hooks never fire and the params stay
on the wrong device, giving a `cuda:0 vs cpu` mismatch under partial load. The correct
fix is the declarative `self.disable_offload = True` (the audio VAEs do the same),
**not** manual `.to(device)` in the node. Use it only when you've confirmed the raw
read; otherwise let offload manage weights.

## 5. Decode-only / asymmetric AEs

Some checkpoints ship only a usable decoder (Cube3D's encoder is point-cloud-based and
not needed for text-to-3D). That's fine:

- Port only the decode path; load the full state dict with `strict=False` so the
  unused encoder weights are ignored.
- Provide only a decode node; there is no symmetric encode node when there's no encode
  path. (Don't invent a `VAEEncode<Model>` that can't work.)

## 6. Output modality → node

| Modality | Decode node |
|----------|-------------|
| image/video | reuse `VAEDecode` / `VAEDecodeTiled` (returns IMAGE) |
| audio | reuse `VAEDecodeAudio` |
| 3d voxel | `VAEDecodeHunyuan3D` → VOXEL, then `VoxelToMesh` |
| 3d mesh | `VAEDecodeCube` → MESH (marching cubes inline) |
| 3d splat | `VAEDecodeTripoSplat` → SPLAT |

3D currently uses **dedicated** decode nodes because their output socket types and
options differ. Whether they can be consolidated is an open question — see
[investigations/3d-vae-node-consolidation.md](../investigations/3d-vae-node-consolidation.md).

Next: [guides/04-latent-format.md](04-latent-format.md).
