# 00 — Getting started

Before writing any code, do the homework that makes the rest of the port mechanical.

## 1. Get the authoritative reference implementation

Clone the model's upstream repo and read its inference path top to bottom. You are
porting **inference**, not training — ignore the training/encoder paths unless the
graph needs them. Identify, concretely:

- The entry point that turns a prompt/image into output (e.g. upstream's
  `Engine.run` / `pipeline.__call__`).
- Every model component and how they are wired (text encoder → main model → VAE →
  post-processing).
- The exact dtypes and autocast regions used on the forward path. Parity bugs are
  very often "we ran conditioning in bf16 but upstream did it in fp32."
- Any constants that are **not** in the state dict (rope theta, normalization eps,
  bounds, scale factors). These must be hardcoded with a comment saying so.

## 2. Spelunk the checkpoint (state-dict keys)

The checkpoint's keys are your detection signature and your dimension source. Dump
them:

```python
import comfy.utils
sd = comfy.utils.load_torch_file("model.safetensors")
for k in sorted(sd):
    print(k, tuple(sd[k].shape))
```

Look for:

- **Unique keys** that no other supported model has — these become your detection
  predicate (see [guides/01-model.md](01-model.md)). Prefer two or three keys
  together over one generic one.
- **Shapes you can infer dimensions from** so you never hardcode widths/heads/layers.
  Examples from the Cube3D port:
  - `n_embd = wte.weight.shape[1]`
  - `n_layer = count_blocks(keys, "transformer.dual_blocks.{}.")`
  - `n_head = n_embd // dual_blocks.0.attn.pre_x.q_norm.weight.shape[0]`
- Whether the text encoder is **bundled** in the checkpoint or expected to be
  loaded separately.

> Match upstream module/parameter names exactly in your port so the state dict
> loads with no key remapping. This is the single biggest time-saver — see how
> `comfy/ldm/cube/vae.py` mirrors `embedder.weight`, `bottleneck.block.*`,
> `decoder.*`, `occupancy_decoder.*`.

## 3. Find the closest existing native model

You are almost never the first. Find the nearest precedent and copy its structure:

| If your model is… | Look at |
|-------------------|---------|
| A diffusion/flow image model | Flux, SD3, an existing DiT in `comfy/ldm/` |
| A video model | WAN (`comfy/ldm/wan/`) |
| An **autoregressive** token model | Causal-WAN (`comfy/ldm/wan/ar_model.py`, `sample_ar_video`) and Cube3D |
| A 3D shape model | Hunyuan3Dv2 (`comfy/ldm/hunyuan3d/`), Cube3D (`comfy/ldm/cube/`), TripoSplat |
| An audio model | ACE-Step / the audio VAEs in `sd.py` |

Mirroring an existing port means reviewers recognize the shape of your change, and
you inherit decisions that were already litigated.

## 4. Set up a parity harness early

The goal of a native port is **bit-for-bit (or as close as possible) parity** with
upstream under identical inputs. Stand up a harness that loads both upstream and your
port in the same process and compares intermediate tensors (per-layer logits, VAE
latents, final output). See [checklists/parity-validation](../checklists/pre-pr-checklist.md).

Catching a divergence at "layer 3 attention output" is an afternoon; catching it at
"the final mesh looks wrong" is a week.

## 5. Branch + commit hygiene

- Work on a `feat/<model>-native` branch.
- Keep the first commit a working end-to-end skeleton, then make each follow-up
  commit a single, reviewable fix with a message that explains *why* (the Cube3D
  history is a good model: each commit names the trap it closes).
- Every deviation from convention gets a code comment, a line in the commit
  message, **and** an explicit call-out in the PR body/comments where applicable — so
  human review and continued work are fully informed.

Next: [guides/01-model.md](01-model.md).
