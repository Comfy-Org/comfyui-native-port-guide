# Investigation — Can the per-model 3D VAE decode nodes be consolidated?

**Status:** open / analysis only. No code changed.

**Question:** Each native 3D model ships its own VAE decode node. Can they be unified
into one (or fewer) generic node(s) so a new 3D port doesn't have to add yet another
`VAEDecode<Model>`?

## Current state

Three native 3D models, three dedicated decode nodes:

| Node | `vae.decode` options | Output socket | Post-decode step |
|------|----------------------|---------------|------------------|
| `VAEDecodeHunyuan3D` | `num_chunks`, `octree_resolution` | `VOXEL` | separate `VoxelToMesh` / `VoxelToMeshBasic` |
| `VAEDecodeCube` | `resolution_base`, `chunk_size` | `MESH` | marching cubes **inline** in the node |
| `VAEDecodeTripoSplat` | `num_gaussians`, `seed` | `SPLAT` | none (splat is the output) |

The generic `VAEDecode` (in `nodes.py`) is **not** an option for any of them: it hard
-codes `RETURN_TYPES=("IMAGE",)` and calls `vae.decode(latent)` with no `vae_options`.

## Why they diverge

Two independent axes of difference:

1. **Output socket type differs** — `VOXEL` vs `MESH` vs `SPLAT`. ComfyUI type-checks
   sockets, so a single node with a fixed output type can't serve all three, and
   downstream nodes (`VoxelToMesh`, `SaveGLB`, splat viewers) expect specific types.
2. **Options differ** — each model exposes different decode knobs with different
   ranges/defaults and tooltips.

There's also a **post-processing inconsistency**: Hunyuan3D returns a raw `VOXEL` and
defers meshing to a separate `VoxelToMesh` node, while Cube3D does marching cubes
*inside* its decode node and returns a `MESH` directly. So "decode" doesn't even mean
the same thing across models.

## What the v3 IO schema makes possible

`comfy_api/latest/_io.py` does offer dynamic typing primitives that a consolidation
could use:

- `MultiType` — an input/output that advertises several types (e.g. `"VOXEL,MESH,SPLAT"`).
- `MatchType` — generic socket whose type is bound by what's connected.
- `DynamicOutput` / `DynamicCombo` — outputs/combos that change based on inputs.
- `NodeOutput.expand` — return a graph fragment at runtime.

So a polymorphic 3D decode node is **technically feasible**. The question is whether
it's a net win.

## Options

### A. Status quo — keep dedicated nodes
- **Pros:** each node is simple, strongly typed, self-documenting; matches the current
  convention; zero risk. Adding a 3D model = add one small node (cheap).
- **Cons:** N models → N nodes; some boilerplate duplication.
- **Verdict:** the honest baseline. The duplication is small and the nodes are clear.

### B. Unify the "decode" half only (raw decode → a common 3D latent/grid type)
- Introduce a single `VAEDecode3D` that returns a generic intermediate (e.g. a tagged
  "3D field/voxel" type via `MultiType`/`MatchType`), then keep small per-modality
  converter nodes (`VoxelToMesh`, `GridToMesh`, `SplatFromField`).
- This also **fixes the post-processing inconsistency** by making Cube3D defer meshing
  to a converter node like Hunyuan3D already does (marching cubes moves out of the
  decode node into a `GridToMesh`/`MarchingCubes` node).
- **Pros:** one decode node; meshing/visualization become reusable, composable
  converters; consistent mental model ("decode → field → convert").
- **Cons:** requires a shared intermediate type and a `MultiType`/dynamic-output node;
  changes the Cube3D graph (extra converter node) — i.e. *more* nodes in the cube
  graph, not fewer; needs the model's options threaded generically.
- **Verdict:** the most principled direction; biggest consistency payoff; medium effort.

### C. Fully polymorphic single node (decode + convert, dynamic output)
- One node that dispatches entirely on VAE type, dynamically setting options and output
  socket (`DynamicCombo` for options + dynamic output type).
- **Pros:** truly one node for everything.
- **Cons:** highest complexity; dynamic outputs/options are harder to validate and to
  reason about in the graph; per-model option sets still have to live somewhere; the
  "one big node that behaves differently per input" pattern is generally discouraged.
- **Verdict:** likely over-engineered; rejected unless B proves insufficient.

## Recommendation (for discussion)

Pursue **Option B** if/when consolidation is prioritized, framed as *consistency*
rather than *node count*:

1. Define a shared 3D field/voxel intermediate type (or reuse `VOXEL` generalized).
2. `VAEDecode3D` (single node, options via the VAE's declared schema or `DynamicCombo`)
   returns that intermediate.
3. Small converter nodes: `VoxelToMesh` (exists), a `GridToMesh`/`MarchingCubes`
   converter (extract from `VAEDecodeCube`), and a splat converter.
4. Migrate Cube3D to defer meshing to the converter (aligns it with Hunyuan3D).

Net effect: **one** decode node for all current and future 3D shape models, plus a
library of reusable converters — at the cost of a slightly longer cube graph and a new
shared type. Keep the dedicated nodes (Option A) until B is actually built; don't break
existing graphs.

## Open questions before building

- Is a single shared intermediate 3D type acceptable across voxel/grid/splat, or do
  splats want their own path? (Splat ≠ surface; may not belong with mesh-producing
  decoders at all.)
- Does threading per-model options through a generic node hurt discoverability vs
  named nodes with tailored tooltips?
- Backward compatibility: existing workflows reference `VAEDecodeHunyuan3D` /
  `VAEDecodeTripoSplat` `node_id`s — a consolidation must keep those working
  (deprecate, don't remove).

## Pointers

- `comfy_extras/nodes_hunyuan3d.py` — `VAEDecodeHunyuan3D`, `VoxelToMesh(Basic)`
- `comfy_extras/nodes_cube.py` — `VAEDecodeCube` (inline marching cubes)
- `comfy_extras/nodes_triposplat.py` — `VAEDecodeTripoSplat`
- `comfy_extras/nodes_save_3d.py` — `pack_variable_mesh_batch`, `SaveGLB`
- `comfy_api/latest/_io.py` — `MultiType`, `MatchType`, `DynamicOutput`, `DynamicCombo`
- `nodes.py` — generic `VAEDecode` (`RETURN_TYPES=("IMAGE",)`)
