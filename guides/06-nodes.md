# 06 — Nodes

User-facing nodes live in `comfy_extras/nodes_<model>.py` and use the **v3 IO schema**
(`comfy_api.latest.IO`). The goal is to add the *fewest* new nodes — reuse the graph
that already exists and only contribute the pieces unique to your model.

## Reuse the graph

Before writing a node, check whether an existing one does the job:

| Need | Reuse |
|------|-------|
| Load the main model | `UNETLoader` / `CheckpointLoaderSimple` (via state-dict detection) |
| Load the VAE | `VAELoader` |
| Load + encode text | `CLIPLoader` + `CLIPTextEncode` |
| CFG / guidance | `CFGGuider` |
| Run sampling | `SamplerCustomAdvanced` + `BasicScheduler` + `RandomNoise` |
| Save 3D | `SaveGLB` |
| Save image/audio | `SaveImage` / `SaveAudio` |

The Cube3D port adds only four nodes (`EmptyCubeLatent`, `CubeCodebookPatch`,
`SamplerCube`, `VAEDecodeCube`) and reuses everything else.

## v3 IO node skeleton

```python
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO

class VAEDecodeCube(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeCube",
            display_name="VAE Decode Cube (3D)",
            category="latent/3d",
            inputs=[
                IO.Vae.Input("vae"),
                IO.Latent.Input("samples"),
                IO.Float.Input("resolution_base", default=8.0, min=4.0, max=10.0, step=0.5),
                IO.Int.Input("chunk_size", default=100000, min=1000, max=2000000, advanced=True),
            ],
            outputs=[IO.Mesh.Output()],
        )

    @classmethod
    def execute(cls, vae, samples, resolution_base, chunk_size) -> IO.NodeOutput:
        grid = vae.decode(samples["samples"],
                          vae_options={"resolution_base": resolution_base, "chunk_size": chunk_size})
        # … marching cubes per batch → MESH …
        return IO.NodeOutput(mesh)

class CubeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [EmptyCubeLatent, CubeCodebookPatch, SamplerCube, VAEDecodeCube]

async def comfy_entrypoint() -> CubeExtension:
    return CubeExtension()
```

Register the module in `nodes.py` (the import list) so the extension loads.

## Node conventions

- **`node_id`** is stable API — don't rename it later. `display_name` is the UI label.
- Put advanced/expert knobs behind `advanced=True`.
- Write **tooltips** that state the contract (e.g. "Must match the tokenizer: 1024
  for v0.5, 512 for v0.1").
- Use the right **output socket type** (`IO.Image`, `IO.Mesh`, `IO.Voxel`,
  `IO.Splat`, `IO.Latent`, `IO.Model`, …). The graph type-checks these.
- Keep heavy logic in `comfy/ldm/<model>` and call into it; the node should be thin.

## Model-patch nodes

When a model needs a load-time transform that upstream does implicitly, expose it as
a `ModelPatcher` object-patch node so it composes with loading/offload rather than
mutating weights eagerly. Cube3D injects the projected VQ codebook into the GPT's
token-embedding table this way:

```python
class CubeCodebookPatch(IO.ComfyNode):
    @classmethod
    def execute(cls, model, vae) -> IO.NodeOutput:
        gpt = model.get_model_object("diffusion_model")
        codebook = vae.first_stage_model.bottleneck.block.get_codebook()
        proj = gpt.shape_proj(codebook.to(gpt.shape_proj.weight))
        old = model.get_model_object("diffusion_model.transformer.wte.weight")
        new = old.clone()
        new[:proj.shape[0]] = proj.to(new)   # only overwrite the code rows; keep special-token rows
        m = model.clone()
        m.add_object_patch("diffusion_model.transformer.wte.weight",
                           torch.nn.Parameter(new, requires_grad=False))
        return IO.NodeOutput(m)
```

Key points: `model.clone()` + `add_object_patch(...)` (never mutate in place), and
only overwrite the rows you mean to.

## Output types & 3D

3D output types (`VOXEL`, `MESH`, `SPLAT`) are distinct sockets, which is why each 3D
model has its own decode node today.

Next: [guides/07-optimizations.md](07-optimizations.md).
