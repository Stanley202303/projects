---
name: onshape-collision-simulation
description: Diagnose, implement, and validate realistic Onshape-driven collision, deformation, perforation, fragmentation, CFD-coupled motion, and ParaView export workflows. Use when simulations show phasing, incorrect placement, unrealistic dents/fragments, mass loss, missing geometry, invalid VTK fields, API overuse, or need strict physics regression tests.
---

# Onshape Collision Simulation

Use this procedure for simulations built from Onshape assemblies, BOM/material data, STL meshes, OpenFOAM loads, moving bodies, and ParaView output. Preserve simple, testable numerical methods unless a solver upgrade is explicitly requested.

## Establish evidence first

1. Read `AGENTS.md`; identify the CLI entry point, runner, Onshape client, motion/collision solver, structural solver, exporter, and tests.
2. Preserve unrelated dirty files and generated artifacts. Never reset the worktree.
3. Inspect supplied movies with `ffprobe` and `ffmpeg`, extracting frames around first contact, maximum deformation, fragmentation, and late motion.
4. Inspect the matching VTK/VTP frame and logs. Quantify patch IDs, triangle counts, bounds, masses, velocities, and stress fields. Decide whether an apparent duplicate is solver geometry, fragment geometry, or a ParaView association problem.

## Control Onshape requests

- Normalize document/workspace/element URLs and include export/configuration parameters in cache keys.
- Cache successful `GET` responses (STL, BOM, assembly, mates, mass properties) atomically on disk. Never cache `POST` requests or failures.
- Use a finite TTL for workspace URLs and provide explicit cache refresh/disable controls.
- For HTTP 429, honor `Retry-After`, back off, and cap retries. For HTTP 402, stop retrying: the annual quota is exhausted. Reduce calls, use cache, or contact Onshape.
- Keep credentials out of source control and logs.

## Preserve assembly and material truth

1. Import every occurrence using its Onshape world transform. Do not recenter or independently reorient child solids.
2. Detect mates, including suppressed mates. Suppressed mates behave as absent; disconnected unmated solids are independent six-DOF bodies.
3. Read the BOM before selecting density, mass, Young’s modulus, Poisson ratio, yield strength, failure strain, thickness, and fracture behaviour. Do not replace valid BOM values with name-based defaults.
4. Validate the initial gap, orientation, world-space bounds, straight approach axis, and velocity before the first dynamic step.

## Collision and structural invariants

- Run swept collision detection before discrete overlap correction, then repeatedly recheck all active bodies—including fragments—after every correction, rotation, constraint, deformation, and environment contact.
- Keep anchored/constrained targets immobile where required; remove only forbidden motion components. Resolve closing normal velocity with material-dependent impulse/restitution.
- Never hide tunnelling by deleting geometry or teleporting bodies. Separate, deform/rotate, update velocity, and iterate until no overlap remains or a strict convergence limit reports failure.
- Keep mass and momentum explicit. On detachment, subtract exactly the triangle mass from the parent and carry fragment velocity. If scatter is clamped for a supported target, document the support reaction; for free parents, transfer the opposite momentum.
- Use a stable explicit shell/MPM-style update with CFL/substepping and material stiffness. Avoid arbitrary global displacement multipliers that make plates thicken, split, or recover after permanent damage.
- Model thin closed STL plates on one physical midsurface. Do not evolve front and back faces as independent shells.

## Fragmentation

1. Derive the physical hole radius from impact energy, material strength, thickness, and projectile geometry.
2. Ductile materials should eject one coherent local plug/rim. Do not use contact-radius or plate-thickness lower bounds that create large sectors.
3. Brittle materials may crack/chip beyond the hole, but use a bounded material-derived radius and controlled fragment count.
4. Give every emitted triangle exactly one fragment owner. Never mark triangles emitted before creating a visible body, and never emit one twice.
5. Build fragment shape from reference geometry plus mean rigid displacement. Do not use unstable highly strained shell nodes as the detached outline.
6. Retain mass and velocity. Normal-impact fragments should travel downstream with bounded transverse scatter; oblique fragments should follow the impact axis. Test speed, direction, size, mass, and ownership numerically.
7. Retain a non-empty structural carrier when all coarse elements touch the hole, so the next step cannot call `min()`/`max()` on an empty mesh.

## ParaView and stress output

- Write a PVD after every completed dynamic step so a later failure leaves a usable partial result.
- Use duplicated triangle vertices for robust previews. Store true element quantities as one value per polygon in VTK `CellData`; use `PointData` only for fields intended to interpolate.
- Validate every PointData tuple count against points and every CellData tuple count against polygons. Never reintroduce `CpPanel`/`CpAbsPanel` with mismatched lengths.
- Export velocity, structural displacement, perforation/failed-element flags, and per-triangle plane-stress von Mises stress, stress-to-yield ratio, and yielded flag using BOM properties.
- If a movie shows two surfaces, inspect patch IDs and VTK cell counts before changing physics; it may be duplicate front/back extrusion or an association error.

## Strict validation

Run focused tests first, then the full suite with plugin autoload disabled when needed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_collision_convergence.py -q --disable-warnings
```

Require tests for:

- straight initial approach, world placement, mates, suppressed mates, and disconnected-body independence;
- no phasing under translation/rotation, repeated collision convergence, and anchored targets;
- dent persistence and evolving perforation;
- ductile plug radius/size/count and brittle wider fracture;
- fragment speed, direction, continued motion, nearby-body effects, and rebound;
- exact triangle ownership, mass conservation, and free-parent momentum reaction;
- non-empty parent geometry after complete coarse fragmentation;
- analytical per-triangle von Mises stress and VTP CellData lengths;
- partial PVD creation after each step;
- cache hit, refresh, TTL expiry, POST non-caching, and 429 handling.

Use analytical expectations wherever possible: zero stress when undeformed, constant-strain triangle stress, tight mass sums, no duplicate triangle IDs, and no unresolved overlaps. Regenerate the movie after tests; an old movie cannot validate a new fix.

## Common mistakes

- Claiming a visual fix without rerunning the simulation.
- Treating stale VTK/PVD or cache data as current output.
- Splitting a closed plate into front/back shell systems.
- Using material-independent fragment radii or arbitrary fragment counts.
- Using distorted current shell nodes as the detached fragment outline.
- Dropping tiny fragments, filtering them from export, or double-counting mass.
- Updating only original components so fragments stop or miss CFD/collisions.
- Deleting VTK fields instead of fixing association and tuple lengths.
- Performing one collision pass instead of rechecking after every state change.
- Treating skew warnings as proof of physical correctness.
- Retrying Onshape 402 quota failures or attempting to bypass limits with extra credentials.
