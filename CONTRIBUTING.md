# Contributing

This is a living, documentation-only repo. Contributions that improve guidance or add
worked examples are welcome.

## What to contribute

- **New case studies.** A new native port is the best contribution — document it like
  [case-studies/cube3d.md](case-studies/cube3d.md): the graph, what changed in core,
  the traps it hit (mapped to guide sections), and parity results.
- **Corrections.** ComfyUI core moves. If a guide references an API/path that changed
  on current `master`, fix it and note the date/commit you verified against.
- **New patterns.** If a port reveals a convention this guide doesn't cover, add it to
  the relevant guide and to [reference/conventions.md](reference/conventions.md).
- **Investigations.** Open questions (like the 3D node consolidation) belong in
  `investigations/` as analysis docs, clearly marked as proposals, not decisions.

## Style

- Ground every claim in real code: cite `file` / `class` / `node_id` names. Prefer a
  short real snippet from an existing port over invented pseudocode.
- Keep guides task-focused and skimmable (tables, decision trees, short sections).
- Link between guides instead of repeating; each fact has one home.
- When you state a parity result, include the tolerance and the hardware/torch build.

## Verifying against core

Before asserting "ComfyUI does X", check it in a current checkout
(`comfy/`, `comfy_extras/`, `comfy_api/`). Cite the file. If you can't verify, mark the
claim as unverified rather than implying certainty.

## The golden rule (applies to ports, restated for docs)

Every recommendation that deviates from a ComfyUI convention must explain **why** —
the same standard the guide asks of port authors. See
[reference/conventions.md](reference/conventions.md).
