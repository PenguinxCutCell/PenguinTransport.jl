# PenguinTransport.jl

[![CI](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl)

`PenguinTransport.jl` is a cut-cell scalar transport package built on:
- `CartesianGeometry.jl`
- `CartesianOperators.jl`
- `PenguinBCs.jl`
- `PenguinSolverCore.jl`

## Governing Equation

The package targets scalar transport in conservative form:

$$
\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u}\,\phi) = s
$$

For steady problems:

$$
\nabla \cdot (\mathbf{u}\,\phi) = s
$$

## Numerical Schemes

Spatial advection discretization:
- `Centered()`
- `Upwind1()`

Time integration for unsteady solves (`solve_unsteady!`):
- `:BE` (Backward Euler, $\theta = 1$)
- `:CN` (Crank-Nicolson, $\theta = 1/2$)
- numeric `theta` values are also accepted

## Embedded Interface Convention

- Outer box boundaries use advection BC types (`Inflow`, `Outflow`, `Periodic`).
- Embedded interface (`Γ`) uses sign-based closure with `s = u_\gamma \cdot n_\gamma`:
  - if `s < 0` and `bc_interface` provides a value, inflow Dirichlet is imposed on interface unknowns (`Tγ = g`),
  - otherwise (`s >= 0`, or no inflow value provided), continuity closure is used (`Tγ = Tω`).
- No-flow interface mode is recovered by setting interface velocity to zero (`uγ = 0`), e.g. in 2D: `uγ = (zeros(nt), zeros(nt))`.

## Feature Status

| Area | Item | Status | Notes |
|---|---|---|---|
| Models | Monophasic steady transport | Implemented | `TransportModelMono` + `assemble_steady_mono!` + `solve_steady!` |
| Models | Monophasic unsteady transport | Implemented | `assemble_unsteady_mono!` + `solve_unsteady!` |
| Models | Two-phase steady transport | Implemented | `TransportModelTwoPhase` + `assemble_steady_two_phase!` + flux continuity coupling |
| Models | Two-phase unsteady transport | Implemented | `assemble_unsteady_two_phase!` + `solve_unsteady!` + theta-method |
| Advection space scheme | Centered | Implemented | `Centered()` |
| Advection space scheme | First-order upwind | Implemented | `Upwind1()` |
| Time scheme | Backward Euler | Implemented | `scheme=:BE` |
| Time scheme | Crank-Nicolson | Implemented | `scheme=:CN` |
| Time scheme | Generic theta method | Implemented | Numeric `scheme` accepted as `theta` |
| Source term | Constant or callable source | Implemented | scalar/callback `(x...)` or `(x..., t)` |
| Outer BCs | Inflow / Outflow / Periodic | Implemented | Handled through `PenguinBCs.jl` border conditions |
| Embedded interface BC | No-flow boundary (`u·n=0` on `Γ`) | Implemented | Use zero interface velocity input (`uγ = 0`) |
| Embedded interface BC | Embedded inflow/outflow scalar imposition | Implemented | Sign-based on `uγ·nγ`: inflow uses `bc_interface`, else continuity |

## Current Limitation

- Two-phase interface configuration with both phases locally inflow at the same `Γ` cell is treated as ill-posed and raises an `ArgumentError`.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/PenguinxCutCell/PenguinTransport.jl")
```

## Quick Start

```julia
using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

grid = (0.0:0.05:1.0, 0.0:0.05:1.0)
cap = assembled_capacity(full_moments(grid); bc=0.0)
nt = cap.ntotal

bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
uω = (ones(nt), zeros(nt))
uγ = (ones(nt), zeros(nt))

model = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Centered())
result = solve_unsteady!(model, zeros(nt), (0.0, 1.0); dt=0.01, scheme=:CN)
```

## Examples

See runnable scripts in `examples/`:
- `smooth_blob_translation.jl`
- `sharp_peak_advection.jl`
- `manufactured_solution.jl`
- `embedded_interface_bc_validation.jl`
- `two_phase_planar_1d_validation.jl`
- `two_phase_2d_planar_sanity.jl`

## Documentation

Local docs are built with Documenter.jl from the `docs/` folder:

```bash
julia --project=docs docs/make.jl
```
