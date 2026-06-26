# 04 — Latent format

`comfy/latent_formats.py` declares the shape and scaling of your model's latent so the
sampler plumbing, empty-latent nodes, and VAE all agree. Getting the layout wrong
causes silent truncation, not a crash — so this deserves care.

## The class

```python
class Cube3D(LatentFormat):
    # Cube3D's "latent" is a flat sequence of VQ token IDs (one scalar per position),
    # so it maps to a channels-first 1D latent (B, 1, num_tokens), mirroring
    # Hunyuan3Dv2's (B, C, L) convention. latent_channels=1 keeps
    # fix_empty_latent_channels from truncating the token sequence. scale_factor=1.0
    # since IDs must pass through process_latent_in/out unchanged.
    latent_channels = 1
    latent_dimensions = 1
    scale_factor = 1.0
```

| Field | Meaning |
|-------|---------|
| `latent_channels` | size of the channel axis (dim=1) |
| `latent_dimensions` | number of spatial dims (1 for sequence, 2 for image, 3 for video/volume) |
| `scale_factor` / `latent_rgb_factors` | scaling applied in `process_latent_in/out`; `1.0` = pass-through |

Wire it into `supported_models.<Model>.latent_format`.

## The `fix_empty_latent_channels` trap

`comfy.sample.fix_empty_latent_channels` runs on **every** empty latent. When
`latent_format.latent_channels != latent.shape[1]`, it calls
`repeat_to_batch_size(..., dim=1)`, which **narrows (truncates)** the channel axis to
`latent_channels`.

This is the trap that drove the Cube3D latent design. Three layouts were considered
for a sequence of 1024 token IDs:

| Layout | What happens | Verdict |
|--------|--------------|---------|
| `(B, num_tokens, 1)` — tokens in channels | `fix_empty_latent_channels` narrows dim=1 from 1024 → `latent_channels` | **broken** (truncates the sequence) |
| `(B, num_tokens)` — true 2D + a core `samplers.py` guard | still truncates; also needs a core edit | **avoid** (wider blast radius) |
| `(B, 1, num_tokens)` — channels-first 1D | `shape[1] == latent_channels == 1` → no truncation; `ndim == 3` so `encode_model_conds` sees a valid `noise.shape[2]` | **chosen** — zero core sampler changes |

Lesson: **put your sequence/spatial size in the trailing dims and keep the channel
axis (dim=1) equal to `latent_channels`.** Mirror an existing model with the same
dimensionality — Cube3D and Hunyuan3Dv2 both use channels-first `(B, C, L)`.

## The empty-latent node

Provide an `Empty<Model>Latent` node that emits the correct shape on the
`intermediate_device`:

```python
latent = torch.zeros([batch_size, 1, num_tokens],
                     device=comfy.model_management.intermediate_device())
return IO.NodeOutput({"samples": latent, "type": "cube_tokens"})
```

Match the dims here to your `LatentFormat`. If the sampler only needs the *length*,
the contents being zeros is fine — the sampler overwrites them.

## Discrete (token-ID) latents

If your latent holds **token IDs** rather than continuous values:

- `scale_factor = 1.0` and pass-through `process_latent_in/out` (don't rescale IDs).
- The sampler/VAE is responsible for `.round().long().clamp(...)` before lookup —
  the latent itself stays float to flow through the standard plumbing.

Next: [guides/05-sampler.md](05-sampler.md).
