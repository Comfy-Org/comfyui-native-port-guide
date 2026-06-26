# ComfyUI Native Port Guide

Guidelines, conventions, and worked examples for adding **native** support for new
models to [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — that is, support
that lives in core (`comfy/`, `comfy_extras/`) and reuses ComfyUI's model
management, ops, conditioning, sampling, and node systems, rather than a bolt-on
custom-node pack that re-implements its own loading and offload.

This repo is documentation only. It distills the patterns that recur across real
native ports (Cube3D, Hunyuan3Dv2, TripoSplat, Causal-WAN, the audio VAEs, …) into
reusable guidance so the next port is faster, more consistent, and easier to review.

## What "native" means here

A native port:

- Loads through the **standard loaders** (`UNETLoader`/`CheckpointLoader`,
  `VAELoader`, `CLIPLoader`) by being **detected from its state-dict keys** — no
  bespoke loader node.
- Uses **`comfy.ops`** for its layers so streaming offload, weight casting, and
  low-VRAM paths work for free.
- Plugs into **model management** (the `ModelPatcher`/offload system) instead of
  manually moving weights to a device.
- Reuses existing **conditioning, guiders, samplers, and output nodes** wherever
  possible; only adds new nodes/samplers when the architecture genuinely requires it.
- **Documents every deviation** from these conventions in-code, with a reason.

If you find yourself calling `load_models_gpu(...)` + `.to(device)` by hand inside a
node, writing your own loader, or re-implementing attention without `comfy.ops`, you
are probably fighting the system — see [reference/conventions.md](reference/conventions.md).

## The eight touchpoints

Almost every native port touches some subset of these. Each has its own guide.

```
comfy/ldm/<model>/…          ported model code (must use comfy.ops)   guides/01-model.md
comfy/model_detection.py     detect arch from unique state-dict keys  guides/01-model.md
comfy/supported_models.py    BASE subclass: latent_format, clip_target guides/01-model.md
comfy/model_base.py          BaseModel subclass: extra_conds          guides/01-model.md
comfy/latent_formats.py      latent channels / dims / scale_factor    guides/04-latent-format.md
comfy/text_encoders/…        text encoder port (or reuse existing)    guides/02-text-encoder.md
comfy/sd.py                  VAE detection + construction             guides/03-vae.md
comfy/k_diffusion/sampling.py custom sampler (only if not diffusion)  guides/05-sampler.md
comfy_extras/nodes_<model>.py user-facing nodes                       guides/06-nodes.md
```

## Decision tree

```
Is the new model weight-compatible with an existing arch?
├─ yes → extend the existing supported_models entry; you may need nothing else.
└─ no  → add a detection branch keyed on UNIQUE state-dict keys.

Does it generate via iterative denoising (diffusion/flow)?
├─ yes → reuse KSampler / SamplerCustomAdvanced; no custom sampler needed.
└─ no  → (autoregressive, single-shot, etc.) add a dedicated sampler and make
         BaseModel._apply_model raise a clear error.  guides/05-sampler.md

Does it ship its own text encoder?
├─ yes → port it under comfy/text_encoders and return it from clip_target().
└─ no  → return None from clip_target() and reuse an existing CLIP loader.  guides/02-text-encoder.md

What is the output modality?
├─ image/video      → reuse VAEDecode / VAEDecodeTiled.
├─ audio            → reuse VAEDecodeAudio.
└─ 3d (mesh/voxel/splat) → dedicated decode node (current convention).  guides/03-vae.md
```

## Map of this repo

| Path | What it covers |
|------|----------------|
| [guides/00-getting-started.md](guides/00-getting-started.md) | Setup, finding the reference impl, state-dict spelunking |
| [guides/01-model.md](guides/01-model.md) | Porting the model: `comfy/ldm`, detection, `supported_models`, `model_base` |
| [guides/02-text-encoder.md](guides/02-text-encoder.md) | Text encoders: reuse vs port, `clip_target`, `extra_conds`, dtype/padding traps |
| [guides/03-vae.md](guides/03-vae.md) | VAE detection in `sd.py`, managed decode, `process_output`, `working_dtypes` |
| [guides/04-latent-format.md](guides/04-latent-format.md) | Latent layout, `fix_empty_latent_channels` trap, empty-latent nodes |
| [guides/05-sampler.md](guides/05-sampler.md) | When you need a custom sampler; AR / non-diffusion patterns |
| [guides/06-nodes.md](guides/06-nodes.md) | v3 IO nodes, reusing graph nodes, output types |
| [guides/07-optimizations.md](guides/07-optimizations.md) | Offload, casting, attention, memory sizing, perf |
| [reference/conventions.md](reference/conventions.md) | Things to avoid + the "document every deviation" rule |
| [reference/ops-and-model-management.md](reference/ops-and-model-management.md) | `comfy.ops`, ModelPatcher, offload internals |
| [checklists/pre-pr-checklist.md](checklists/pre-pr-checklist.md) | Parity validation + review checklist |
| [case-studies/cube3d.md](case-studies/cube3d.md) | The Roblox Cube3D port, annotated end-to-end |
| [investigations/3d-vae-node-consolidation.md](investigations/3d-vae-node-consolidation.md) | Can the per-model 3D VAE decode nodes be unified? |

## License

[GNU General Public License v3.0](LICENSE), matching ComfyUI. Applies to both the
documentation and any example code in this repo.

## Status

Living document. The Cube3D case study is the most complete worked example; other
ports are referenced where they illustrate a pattern. PRs that add new case studies
or correct guidance against the current ComfyUI `master` are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).
