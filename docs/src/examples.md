# Examples

Instantiate once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run one script:

```bash
julia --project=. examples/smooth_blob_translation.jl
```

## Basic 1D / Periodic Transport

### `examples/smooth_blob_translation.jl`

- Physical meaning: smooth periodic scalar advection.
- Demonstrates: baseline mono unsteady behavior and phase speed.
- API: `TransportModelMono` + `solve_unsteady!`.
- Check: profile translates with expected periodic wrap-around.

## Sharp Transport / Discontinuous Profiles

### `examples/sharp_peak_advection.jl`

- Physical meaning: advection of sharper structures.
- Demonstrates: `Centered()` vs `Upwind1()` tradeoff.
- API: mono fixed geometry.
- Check: upwind is monotone/diffusive, centered is sharper but may oscillate.

## Manufactured Validation

### `examples/manufactured_solution.jl`

- Physical meaning: transport against an analytic target.
- Demonstrates: consistency/convergence trends.
- API: mono fixed assembly/solve paths.
- Check: error decreases with refinement.

## Embedded-Interface Mono Advection

### `examples/embedded_interface_bc_validation.jl`

- Physical meaning: fixed embedded interface with local inflow/outflow switching.
- Demonstrates: closure rule from `s = uγ·nγ`.
- API: `assemble_steady_mono!` / `solve_unsteady!`.
- Check: inflow interface cells impose `Tγ=g`; others use continuity.

## Two-Phase Transport Validations

### `examples/two_phase_planar_1d_validation.jl`

- Physical meaning: planar two-phase transport benchmark.
- Demonstrates: two-phase interface closure with ordering `(ω1, γ1, ω2, γ2)`.
- API: `TransportModelTwoPhase`, steady/unsteady solves.
- Check: phase-wise behavior and interface rows match expected closure mode.

### `examples/two_phase_2d_planar_sanity.jl`

- Physical meaning: 2D two-phase cut-cell sanity case.
- Demonstrates: robust two-phase assembly on 2D geometry.
- API: `solve_unsteady!(TransportModelTwoPhase(...))`.
- Check: bounded stable evolution.

## Moving Mono Examples

### `examples/moving_mono_material_translation.jl`

- Physical meaning: moving embedded body translating with the flow.
- Demonstrates: moving slab assembly with relative interface handling.
- API: `MovingTransportModelMono` + `solve_unsteady_moving!`.
- Check: final field error and qualitative material transport consistency.

### `examples/moving_mono_interface_inflow.jl`

- Physical meaning: moving interface with local relative inflow segments.
- Demonstrates: switching by `λ = (uγ - wγ)·nγ`.
- API: moving mono assembly/solve.
- Check: inflow interface data activates only where `λ < 0`.

## Moving Two-Phase Examples

### `examples/moving_two_phase_planar_translation.jl`

- Physical meaning: translating planar interface with phase fields.
- Demonstrates: moving two-phase assembly and phase-wise error reporting.
- API: `MovingTransportModelTwoPhase` + `solve_unsteady_moving!`.
- Check: phase errors decrease under refinement (positive trend).

### `examples/moving_two_phase_relative_flux_demo.jl`

- Physical meaning: relative interface speed pattern changes by phase/time.
- Demonstrates: closure branch selection from `λ1`, `λ2`.
- API: moving two-phase assembly path.
- Check: printed sign pattern matches expected inflow/outflow mode.

## Verification Map (Tests ↔ Example Families)

- Convergence-oriented:
  - tests: `Convergence order: upwind ≈ 1, centered > 1.5`, moving mesh-trend regressions
  - examples: `manufactured_solution.jl`, `smooth_blob_translation.jl`, `moving_*_translation.jl`
- Interface-closure validation:
  - tests: mono sign-based closure, relative-speed regression (`uγ=wγ => λ=0`)
  - examples: `embedded_interface_bc_validation.jl`, `moving_mono_interface_inflow.jl`
- Two-phase behavior:
  - tests: two-phase row-pattern, both-inflow rejection (fixed/moving)
  - examples: `two_phase_planar_1d_validation.jl`, `moving_two_phase_relative_flux_demo.jl`
