# Reference

## Governing Equation

```math
\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u}\,\phi) = s
```

Steady form:

```math
\nabla \cdot (\mathbf{u}\,\phi) = s
```

## Schemes

Spatial advection:
- `Centered()`
- `Upwind1()`

Temporal integration in `solve_unsteady!`:
- Backward Euler: `scheme=:BE`
- Crank-Nicolson: `scheme=:CN`
- Generic theta: `scheme=<number>`

## Embedded Interface Convention

- Outer box boundaries use `Inflow`/`Outflow`/`Periodic`.
- Embedded interface `Γ` uses sign-based closure with `s = uγ·nγ`:
  - if `s < 0` and `bc_interface` provides a value, inflow Dirichlet is imposed (`Tγ = g`),
  - otherwise (`s >= 0`, or no inflow value) continuity closure is used (`Tγ = Tω`).
- No-flow mode is recovered with zero interface velocity input (`uγ = 0`).

## Feature Status

| Area | Item | Status | Notes |
|---|---|---|---|
| Models | Monophasic steady transport | Implemented | `TransportModelMono` + `assemble_steady_mono!` + `solve_steady!` |
| Models | Monophasic unsteady transport | Implemented | `assemble_unsteady_mono!` + `solve_unsteady!` |
| Advection space scheme | Centered | Implemented | `Centered()` |
| Advection space scheme | First-order upwind | Implemented | `Upwind1()` |
| Time scheme | Backward Euler | Implemented | `scheme=:BE` |
| Time scheme | Crank-Nicolson | Implemented | `scheme=:CN` |
| Time scheme | Generic theta method | Implemented | Numeric `scheme` accepted as `theta` |
| Outer BCs | Inflow / Outflow / Periodic | Implemented | Through `PenguinBCs.jl` border conditions |
| Embedded interface BC | No-flow boundary (`u·n=0` on `Γ`) | Implemented | Use zero interface velocity input (`uγ = 0`) |
| Embedded interface BC | Embedded inflow/outflow scalar imposition | Implemented | Sign-based on `uγ·nγ`: inflow uses `bc_interface`, else continuity |

## Current Limitation

- Two-phase transport model is not yet implemented.
