# 05 — Samplers

**Default to reusing ComfyUI's samplers.** Only add a custom sampler when the model
does **not** generate by iterative denoising.

## Decision

```
Does generation = iterative denoising over sigmas (diffusion / flow / rectified flow)?
├─ yes → reuse KSampler / SamplerCustomAdvanced. Implement the model as a normal
│        BaseModel with apply_model. You need NO custom sampler.
└─ no  → (autoregressive token generation, single forward, search, etc.)
         add a dedicated sampler in comfy/k_diffusion/sampling.py and expose it
         via a Sampler node + SamplerCustomAdvanced.
```

If you're in the second branch, also make `BaseModel._apply_model` raise (see
[guides/01-model.md](01-model.md)) so nobody wires a KSampler to a non-diffusion model.

## Writing a non-diffusion sampler

A ComfyUI sampler is `sample(model, x, sigmas, extra_args, callback, disable, **opts)`.
For a non-diffusion model you **ignore `x`/`sigmas` as noise** and use them only for
shape/scheduling bookkeeping, then drive the real generation loop yourself.

The Cube3D autoregressive sampler (`sample_cube` in `comfy/k_diffusion/sampling.py`)
is the reference pattern:

- It is explicitly **not** a diffusion sampler: the noised `x` and `sigmas` are
  ignored; only `x.shape` (batch, channels=1, `num_tokens`) is read.
- It reads conditioning directly off the guider: `guider.conds["positive"]` /
  `["negative"]`, and calls `diffusion_model` directly rather than through
  `apply_model`.
- It reproduces upstream's generation loop faithfully — here, autoregressive token
  sampling with linearly-decaying CFG `γ_i = cfg·(T − i)/T`, then
  `logits = (1+γ)·cond − γ·uncond`, with fp32 weights + bf16 autocast on CUDA.
- Token selection: greedy `argmax` when `top_p >= 1` (deterministic, upstream
  default); otherwise top-p nucleus filtering + `torch.multinomial`.
- It returns IDs in the **same latent layout** the format expects (`(B, 1, T)`), and
  repeats conditioning to the latent batch size with
  `comfy.utils.repeat_to_batch_size`.

Register it the same way diffusion samplers are registered (add `"cube"` to the
sampler table consumed by `comfy.samplers.ksampler`).

## The Sampler node

Expose the sampler as a `SAMPLER`-typed node so it composes with
`SamplerCustomAdvanced`, `BasicScheduler`, `RandomNoise`, and `CFGGuider`:

```python
class SamplerCube(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SamplerCube",
            display_name="Sampler Cube (autoregressive)",
            category="sampling/custom_sampling/samplers",
            inputs=[IO.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01)],
            outputs=[IO.Sampler.Output()],
        )

    @classmethod
    def execute(cls, top_p) -> IO.NodeOutput:
        return IO.NodeOutput(comfy.samplers.ksampler("cube", {"top_p": top_p}))
```

This means you reuse the entire custom-sampling graph (guider, scheduler, noise,
`SamplerCustomAdvanced`) and only contribute the sampler itself. For a single-shot
model, drive the schedule with `BasicScheduler(steps=1)`.

## Parity traps for samplers

- **Constants not in the state dict.** `rope_theta`, CFG schedule shape, eps. The
  released Cube3D weights use `rope_theta = 10000` while the upstream dataclass
  default is 1000 — verify against the *released config*, not the default.
- **CFG schedule variants.** Upstream may have multiple engines (e.g. a CUDA-graph
  `EngineFast` that fixes γ at step 0, vs a plain `Engine` that decays from step 0).
  Pick the one that matches the reference output and note which.
- **Autocast region & dtype — match upstream's *boundaries*, not necessarily its
  dtype.** Upstream here means the original research code, which is often heavy-handed
  (e.g. forcing fp32 in places ComfyUI would happily run in bf16/fp16). Don't blindly
  copy that — default to ComfyUI's lighter dtype conventions and let model management
  pick. **Only** pin a heavier dtype (e.g. keep conditioning/setup in fp32) where
  doing otherwise **noticeably degrades or meaningfully changes the output** — verify
  with your parity harness, then pin it *and document why* (it's a deviation). What you
  must reproduce faithfully is *where* the autocast region begins/ends relative to the
  forward, not the exact dtype upstream happened to use. See
  [guides/02](02-text-encoder.md). (Cube3D does pin conditioning to fp32 — but only
  because bf16 there produced a different mesh.)
- **Determinism.** Greedy argmax is deterministic but can flip on near-ties across
  torch/CUDA builds; this shows up as a different-but-valid result. Document it.

Next: [guides/06-nodes.md](06-nodes.md).
