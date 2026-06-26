# 01 — Porting the model

This is the core of a native port: getting the main network to load from the
checkpoint and run. Four files are involved.

```
comfy/ldm/<model>/…          the ported network (uses comfy.ops)
comfy/model_detection.py     detect the arch from state-dict keys
comfy/supported_models.py    register a BASE subclass
comfy/model_base.py          a BaseModel subclass that wires the network in
```

## 1. `comfy/ldm/<model>/` — the network

Create a package `comfy/ldm/<model>/` and port the inference path.

**Always build layers from `comfy.ops`, not raw `torch.nn`.** This is what lets the
offload/cast system relocate and cast weights without your code knowing:

```python
import comfy.ops
ops = comfy.ops.disable_weight_init

class MLP(nn.Module):
    def __init__(self, dim, hidden, dtype=None, device=None):
        super().__init__()
        self.up = ops.Linear(dim, hidden, dtype=dtype, device=device)
        self.down = ops.Linear(hidden, dim, dtype=dtype, device=device)
```

Rules of thumb:

- Use `ops.Linear`, `ops.Conv2d`, `ops.Embedding`, `ops.LayerNorm`, etc. for any
  layer with **loaded weights**.
- Thread `dtype=` and `device=` through every constructor.
- Norms that must run in fp32 for stability can upcast internally
  (`x.float()` → compute → `.type_as(x)`), as `CubeRMSNorm`/`CubeLayerNorm` do —
  this is fine and does not break offload.
- Parameters you read **outside** a hooked `forward` (e.g. a VQ codebook looked up
  directly) won't be relocated by the cast hooks — see
  [guides/07-optimizations.md](07-optimizations.md) and the `disable_offload`
  discussion in [guides/03-vae.md](03-vae.md).

See [reference/ops-and-model-management.md](../reference/ops-and-model-management.md)
for what `comfy.ops` actually does.

## 2. `comfy/model_detection.py` — detection

Add a branch to `detect_unet_config` keyed on **unique** state-dict keys, and infer
every dimension from shapes. Real example (Cube3D):

```python
if '{}shape_proj.weight'.format(key_prefix) in state_dict_keys and \
   '{}lm_head.weight'.format(key_prefix) in state_dict_keys:  # Roblox Cube3D shape GPT
    dit_config = {}
    dit_config["image_model"] = "cube3d"
    n_embd = state_dict['{}transformer.wte.weight'.format(key_prefix)].shape[1]
    dit_config["n_embd"] = n_embd
    dit_config["n_layer"] = count_blocks(state_dict_keys, '{}transformer.dual_blocks.'.format(key_prefix) + '{}.')
    head_dim = state_dict['{}transformer.dual_blocks.0.attn.pre_x.q_norm.weight'.format(key_prefix)].shape[0]
    dit_config["n_head"] = n_embd // head_dim
    dit_config["rope_theta"] = 10000  # not stored in the state dict; upstream's fixed constant
    return dit_config
```

Guidance:

- Pick keys that **no other supported model** has. One generic key (`latent_in`)
  invites false positives; combine two or three.
- Infer dims from shapes (`count_blocks`, `.shape[…]`). Never hardcode a width that
  the checkpoint already encodes.
- Constants not in the state dict (`rope_theta`, eps) get a comment saying so.
- `image_model` (or an equivalent discriminator) is the string `supported_models`
  matches on.

## 3. `comfy/supported_models.py` — registration

Add a `BASE` subclass. It connects detection → base model, latent format, and the
text encoder.

```python
class Cube3D(supported_models_base.BASE):
    unet_config = {"image_model": "cube3d"}
    unet_extra_config = {}
    sampling_settings = {}
    latent_format = latent_formats.Cube3D            # see guides/04
    memory_usage_factor = 1.0
    supported_inference_dtypes = [torch.float32, torch.bfloat16]

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.Cube3D(self, device=device)

    def clip_target(self, state_dict={}):
        # No bundled text encoder: the cube checkpoint is GPT-only. The graph wires a
        # standard CLIPLoader(clip-l)/CLIPTextEncode, so there is no clip_target to build.
        return None
```

Key fields:

- `latent_format` — must match your latent layout ([guides/04](04-latent-format.md)).
- `supported_inference_dtypes` — what the forward can run in. Upstream keeping fp32
  weights + bf16 autocast → `[torch.float32, torch.bfloat16]`.
- `clip_target` — return a `ClipTarget` if you ship a text encoder, else `None` and
  reuse an existing CLIP loader ([guides/02](02-text-encoder.md)).
- `memory_usage_factor` — used by model management to estimate VRAM
  ([guides/07](07-optimizations.md)).

## 4. `comfy/model_base.py` — the BaseModel subclass

Wrap your network in a `BaseModel` subclass. For **diffusion** models this is mostly
boilerplate (pass `unet_model=...` and implement `extra_conds`). For
**non-diffusion** models (autoregressive, single-shot), make `_apply_model` raise so
nobody silently wires a KSampler to it:

```python
class Cube3D(BaseModel):
    """Autoregressive shape GPT. Generation goes through the dedicated `cube` sampler
    (SamplerCustomAdvanced), never KSampler/apply_model."""
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device,
                         unet_model=comfy.ldm.cube.gpt.DualStreamRoformer)

    def _apply_model(self, *args, **kwargs):
        raise RuntimeError(
            "Cube3D is an autoregressive token model. Use the 'cube' sampler "
            "(SamplerCube + SamplerCustomAdvanced), not KSampler.")

    def extra_conds(self, **kwargs):
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out
```

`extra_conds` is how conditioning reaches your forward: it packages each conditioning
tensor in a `comfy.conds.*` wrapper under the key your model/sampler reads
(`c_crossattn`, `y`, `guidance`, …). See [guides/02](02-text-encoder.md) for how the
conditioning tensors get produced.

Next: [guides/02-text-encoder.md](02-text-encoder.md).
