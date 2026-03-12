# Examples

Instantiate the project once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run a script:

```bash
julia --project=. examples/smooth_blob_translation.jl
```

## Basic 1D / Periodic Transport

### `examples/smooth_blob_translation.jl`

- Physical meaning: smooth periodic advection of a transported scalar profile.
- Demonstrates: baseline mono unsteady transport stability and phase-speed behavior.
- API entry point: `TransportModelMono` + `solve_unsteady!`.
- Qualitative check: profile translates around the periodic domain without boundary forcing artifacts.

## Sharp Transport / Discontinuous Profiles

### `examples/sharp_peak_advection.jl`

- Physical meaning: advection of sharper features.
- Demonstrates: `Centered()` vs `Upwind1()` behavior (dispersion vs diffusion tradeoff).
- API entry point: `solve_unsteady!` with selectable spatial advection scheme.
- Qualitative check: upwind remains monotone but diffuses; centered is sharper but may oscillate.

## Manufactured Validation

### `examples/manufactured_solution.jl`

- Physical meaning: transport against a known analytical target.
- Demonstrates: consistency and convergence behavior of discrete transport assembly.
- API entry point: mono steady/unsteady assembly and solve wrappers.
- Qualitative check: error decreases under refinement and expected scheme choices.

## Embedded-Interface Mono Advection

### `examples/embedded_interface_bc_validation.jl`

- Physical meaning: mono transport with embedded-interface inflow/outflow switching.
- Demonstrates: sign-based interface closure using `s = uγ·nγ`.
- API entry point: `assemble_steady_mono!` / `solve_unsteady!` with `bc_interface`.
- Qualitative check: interface inflow locations impose `Tγ=g`; outflow/no-inflow-data locations use continuity closure.

## Two-Phase Transport Validations

### `examples/two_phase_planar_1d_validation.jl`

- Physical meaning: 1D planar two-phase advection exchange.
- Demonstrates: two-phase closure behavior and phase-interface coupling in ordering `(ω1, γ1, ω2, γ2)`.
- API entry point: `TransportModelTwoPhase`, `solve_steady!`, `solve_unsteady!`.
- Qualitative check: interface relation is satisfied and phase fields follow expected inflow-driven behavior.

### `examples/two_phase_2d_planar_sanity.jl`

- Physical meaning: 2D two-phase planar sanity case.
- Demonstrates: robust two-phase assembly on 2D cut geometry.
- API entry point: `solve_unsteady!(TransportModelTwoPhase(...))`.
- Qualitative check: stable bounded evolution and interface-coupling consistency.

## Verification Map (Tests ↔ Examples)

- Convergence / manufactured behavior:
  - tests: `Convergence order: upwind ≈ 1, centered > 1.5`
  - examples: `manufactured_solution.jl`, `smooth_blob_translation.jl`
- Interface-closure validation:
  - tests: `Embedded interface sign-based inflow/outflow BC`, `API contract regressions` mono-closure subset
  - example: `embedded_interface_bc_validation.jl`
- Two-phase behavior and local closure selection:
  - tests: `Two-phase interface row pattern`, `Two-phase both-inflow local configuration throws`
  - examples: `two_phase_planar_1d_validation.jl`, `two_phase_2d_planar_sanity.jl`
