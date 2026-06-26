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

## Equivalent math ≠ identical output (autoregressive models)

For an autoregressive sampler, **output equality is the wrong parity criterion.** Two
implementations that are mathematically identical but differ only in floating-point
*rounding order* can still produce a completely different final artifact: a single
greedy-`argmax` near-tie flips one token, and because each token is fed back in, the
rest of the sequence cascades into a different (but equally valid) result. A diffusion
model averages such noise out over its denoising steps; an autoregressive decode
*amplifies* it. So **verify equivalence at the math/logits level, not by diffing the
final mesh/image/audio.**

The right harness for an AR port:

1. **Isolated op test** — feed identical tensors to the old and new implementation of
   the op you changed and compare directly (no autoregression in the loop).
2. **Logits/first-token test** — compare the GPT's logits for the first generated
   token under identical conditioning. Agreement here proves the forward pass matches;
   divergence *after* the first flip is expected amplification, not a bug.

Do **not** conclude "the port is broken" from a different final artifact alone.

### Worked example — swapping Cube3D's RoPE for the shared Flux kernel

Cube3D's bespoke complex-number RoPE (`torch.polar` / `view_as_complex`) was migrated
to ComfyUI's shared Flux rotary embedding so it benefits from comfy-kitchen's optimized
`apply_rope` kernel (see [checklists/pre-pr-checklist.md](../checklists/pre-pr-checklist.md)
and [guides/07-optimizations.md](07-optimizations.md)). The pairing convention
(adjacent dims), frequencies (`theta=10000`, `2k/dim`), and rotation (`(x0+ix1)·e^{iθ}`)
are identical; the only difference is Flux computes the angles in fp64 before casting to
fp32.

Isolated op test on a 2×4090 box (cube-original complex RoPE vs Flux torch reference vs
the comfy-kitchen kernel actually used at inference):

| compare | fp32 | bf16 |
|---|---|---|
| ck-kernel vs Flux torch-ref | **0.0 (bit-identical)** | ~3.9e-3 |
| cube-original vs migrated | max 1.5e-4 / mean 1.1e-6 | ~1.6e-2 (≈1 bf16 ULP) |

Yet the end-to-end GLB differed by ~4% (3,931,984 → 3,778,836 bytes, reproducible).
Cause: `sample_cube` runs greedy `argmax` under a **bf16 autocast**, so a ~1-ULP RoPE
difference eventually flips one argmax across the 1024-token decode and the mesh
cascades. The migration is **correct** — the kernel is bit-identical to the reference
in fp32 — and the divergence is the expected amplification described above.

Lessons this reinforces:

- The isolated op + bit-identical fp32 kernel check is what proved correctness; the GLB
  diff alone would have been misleading.
- Don't "fix" this by forcing fp32 autocast — the output isn't degraded, just
  different, which matches the dtype guidance above (only pin a heavier dtype when
  quality *meaningfully* changes).
- Per the golden rule, a change like this must be **called out in the PR body**: state
  that the mesh diverges from the pre-migration baseline due to bf16 autoregressive
  amplification of a mathematically-equivalent op, and attach the isolated-op numbers
  so review proceeds fully informed. See [case-studies/cube3d.md](../case-studies/cube3d.md).

Next: [guides/06-nodes.md](06-nodes.md).
