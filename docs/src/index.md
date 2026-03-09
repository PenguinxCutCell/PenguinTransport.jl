# PenguinTransport.jl

`PenguinTransport.jl` provides cut-cell scalar transport tools for Cartesian grids.

## Documentation Pages

- [Docstrings](docstrings.md)
- [API](api.md)
- [Reference](reference.md)

## Governing Equation

```math
\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u}\,\phi) = s
```

Steady form:

```math
\nabla \cdot (\mathbf{u}\,\phi) = s
```

## Schemes Used

Spatial advection schemes:
- `Centered()`
- `Upwind1()`

Unsteady time schemes in `solve_unsteady!`:
- `:BE` (Backward Euler, `theta = 1`)
- `:CN` (Crank-Nicolson, `theta = 1/2`)
- numeric `theta` values are accepted

## Embedded Interface Convention

- Outer box boundaries use advection boundary conditions (`Inflow`, `Outflow`, `Periodic`).
- Embedded interface `Γ` uses sign-based closure with `s = uγ·nγ`:
  - if `s < 0` and `bc_interface` provides a value, impose `Tγ = g` (inflow Dirichlet),
  - otherwise use continuity closure `Tγ = Tω`.
- No-flow mode is recovered by setting interface velocity to zero, e.g. `uγ = (zeros(nt), zeros(nt))`.

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
| Outer BCs | Inflow / Outflow / Periodic | Implemented | Through `PenguinBCs.jl` border conditions |
| Embedded interface BC | No-flow boundary (`u·n=0` on `Γ`) | Implemented | Use zero interface velocity input (`uγ = 0`) |
| Embedded interface BC | Embedded inflow/outflow scalar imposition | Implemented | Sign-based on `uγ·nγ`: inflow uses `bc_interface`, else continuity |

## Current Limitation

- Two-phase interface configuration with both phases locally inflow at the same `Γ` cell is treated as ill-posed and raises an `ArgumentError`.

## Public API

Main exported interface:
- `TransportModelMono`
- `TransportModelTwoPhase`
- `assemble_steady_mono!`
- `assemble_unsteady_mono!`
- `assemble_steady_two_phase!`
- `assemble_unsteady_two_phase!`
- `solve_steady!`
- `solve_unsteady!`
- `update_advection_ops!`
- `rebuild!`

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/PenguinxCutCell/PenguinTransport.jl")
```

## Minimal Usage

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

Example scripts are available in `examples/`:
- `smooth_blob_translation.jl`
- `sharp_peak_advection.jl`
- `manufactured_solution.jl`
- `embedded_interface_bc_validation.jl`
- `two_phase_planar_1d_validation.jl`
- `two_phase_2d_planar_sanity.jl`

## Build Docs

```bash
julia --project=docs docs/make.jl
```
