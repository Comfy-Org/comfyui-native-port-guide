# Checklist — before opening the PR

Work through this before sending a native port for review. Each item links to the
relevant guide.

## Loading & detection

- [ ] Model is detected from **unique** state-dict keys (no false positives against
      existing models). [→01](../guides/01-model.md)
- [ ] All dimensions are **inferred from shapes**, not hardcoded.
- [ ] Module/param names mirror upstream so the checkpoint loads with **no key
      remapping** (`load_state_dict` reports no unexpected/missing keys you didn't
      intend, e.g. decode-only AE encoder weights via `strict=False`).
- [ ] Constants not in the state dict (`rope_theta`, eps, bounds) are **commented**
      as such.

## Ops & model management

- [ ] Every weighted layer uses **`comfy.ops`**. [→07](../guides/07-optimizations.md)
- [ ] No manual `load_models_gpu(...)` / `.to(device)` inside nodes.
      [→03](../guides/03-vae.md)
- [ ] `disable_offload` is set **only** where raw params are read outside a hooked
      forward, with a comment. [→03](../guides/03-vae.md)
- [ ] Load-time weight transforms use `model.clone()` + `add_object_patch`.
      [→06](../guides/06-nodes.md)

## Latent & VAE

- [ ] Latent layout's channel axis equals `latent_format.latent_channels` (no
      `fix_empty_latent_channels` truncation). [→04](../guides/04-latent-format.md)
- [ ] `process_output` is identity if the default `[0,1]` clamp would corrupt output.
      [→03](../guides/03-vae.md)
- [ ] `working_dtypes` reflects real numerical requirements.
- [ ] `memory_used_decode` is sized by the dimension that drives decode memory.
- [ ] VAE decode/encode flows through the managed `comfy.sd.VAE` path.

## Sampling & conditioning

- [ ] Reused KSampler/SamplerCustomAdvanced if diffusion; custom sampler only if not.
      [→05](../guides/05-sampler.md)
- [ ] Non-diffusion `BaseModel._apply_model` **raises** a clear error.
- [ ] Conditioning runs in the dtype/autocast region upstream uses.
      [→02](../guides/02-text-encoder.md)
- [ ] Reused an existing text encoder where possible; `clip_target()` correct.

## Nodes

- [ ] Added the **fewest** new nodes; reused the existing graph.
      [→06](../guides/06-nodes.md)
- [ ] Correct output socket types; stable `node_id`s; useful tooltips.
- [ ] Extension registered in `nodes.py`.

## Dependencies

- [ ] No new pip dependency added for this model alone (vendored small impls instead).
      [→07](../guides/07-optimizations.md)
- [ ] `requirements.txt` is clean (removed anything you stopped using).

## Parity validation

- [ ] Loaded upstream + native in the **same process** and compared intermediate
      tensors (per-layer logits, VAE latents, final output). Recorded the tolerances.
- [ ] Documented any out-of-the-box differences vs upstream (torch/CUDA build, CLIP
      padding, marching-cubes backend, post-processing) as **known and explained**,
      not silent.
- [ ] Re-ran the full graph and confirmed determinism (or documented the source of
      nondeterminism).

## The golden rule

- [ ] **Every deviation from convention is commented in-code with a reason**, and
      called out in the commit message. [→conventions](../reference/conventions.md)

## A good parity harness (sketch)

```python
# Load both stacks in one process, feed identical inputs, compare.
up = load_upstream(...)          # reference impl
nt = load_native(...)            # your comfy.ldm port

x = same_inputs(...)
a = up.forward_capture(x)        # dict of intermediate tensors
b = nt.forward_capture(x)
for k in a:
    d = (a[k].float() - b[k].float()).abs().max().item()
    print(k, "max-abs-diff", d)  # aim for 0.0 on the model forward; record VAE tolerance
```

The Cube3D port hit `max-abs-diff = 0.0` on per-step GPT logits and `3.7e-4` on VQ-VAE
decode latents this way — see [case-studies/cube3d.md](../case-studies/cube3d.md).
