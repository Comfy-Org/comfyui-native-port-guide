# Reference — Conventions: things to do, things to avoid

A condensed checklist of the conventions every native port should follow, and the
anti-patterns that get PRs sent back. Each item links to the guide with the full
rationale.

## The golden rule

> **Document every deviation from convention in-code, with a reason.**

ComfyUI has strong conventions (use `comfy.ops`, route through managed paths, reuse
nodes). When your model genuinely needs to break one, that's allowed — but the
deviation must be **commented at the site** explaining *why*, and called out in the
commit message. A reviewer should never have to guess whether an oddity is intentional.

The Cube3D port's final commit was a pure "document the deviations" pass: it commented
why `working_dtypes` is fp32-only, why `clip_target()` returns `None`, why
`rope_theta=10000`, and removed a dead flag. That is the standard to aim for.

## Do

- ✅ Build weighted layers from **`comfy.ops`** (`ops.Linear`, `ops.Conv2d`, …).
  [→01](../guides/01-model.md) [→07](../guides/07-optimizations.md)
- ✅ **Detect** the model from unique state-dict keys; **infer** dims from shapes.
  [→01](../guides/01-model.md)
- ✅ Mirror upstream **module/parameter names** so the checkpoint loads with no
  remapping. [→00](../guides/00-getting-started.md)
- ✅ **Reuse** existing loaders, text encoders, guiders, samplers, and output nodes.
  [→02](../guides/02-text-encoder.md) [→06](../guides/06-nodes.md)
- ✅ Route VAE decode/encode through the **managed** `comfy.sd.VAE.decode/encode`.
  [→03](../guides/03-vae.md)
- ✅ Pick a latent layout whose channel axis equals `latent_channels`.
  [→04](../guides/04-latent-format.md)
- ✅ Make non-diffusion `BaseModel._apply_model` **raise** a clear error.
  [→01](../guides/01-model.md) [→05](../guides/05-sampler.md)
- ✅ Apply load-time weight transforms via `model.clone()` + `add_object_patch`.
  [→06](../guides/06-nodes.md)
- ✅ Validate against upstream for **parity** and record the tolerance.
  [→checklist](../checklists/pre-pr-checklist.md)

## Avoid

- ❌ **Manual model management** inside a node: `load_models_gpu(...)` + `.to(device)`.
  Route through the managed path instead. [→03](../guides/03-vae.md)
- ❌ **Raw `torch.nn`** weighted layers — breaks offload/casting.
  [→07](../guides/07-optimizations.md)
- ❌ A **bespoke loader node**. Use state-dict detection + standard loaders.
  [→01](../guides/01-model.md)
- ❌ A latent layout that puts your sequence/spatial size in the **channel axis** —
  `fix_empty_latent_channels` will truncate it. [→04](../guides/04-latent-format.md)
- ❌ Leaving `process_output` as the default `[0,1]` clamp when your decode returns
  non-image data (it corrupts signed logits / isosurfaces). [→03](../guides/03-vae.md)
- ❌ `disable_offload = True` as a **shortcut**. Only use it for the real reason
  (raw param reads outside hooked forwards). [→03](../guides/03-vae.md)
- ❌ Adding a **pip dependency** for one model — vendor a small implementation
  instead. [→07](../guides/07-optimizations.md)
- ❌ Computing conditioning in the **wrong dtype / autocast region**.
  [→02](../guides/02-text-encoder.md)
- ❌ Hardcoding dims the checkpoint already encodes. [→01](../guides/01-model.md)
- ❌ Inventing an encode node for a **decode-only** checkpoint.
  [→03](../guides/03-vae.md)
- ❌ Mutating model weights in place instead of using object patches.
  [→06](../guides/06-nodes.md)
- ❌ Renaming a node `node_id` after release (it's stable API).
  [→06](../guides/06-nodes.md)

## Commit & review hygiene

- One reviewable concern per commit; the message says *why*, not just *what*.
- Reference the design thread / parity evidence in the PR description.
- Keep `requirements.txt` clean (remove deps you stopped using).
