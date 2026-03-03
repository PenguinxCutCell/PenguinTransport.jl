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
- Embedded interface `Γ` is handled in no-flow mode (`u·n=0`) by setting interface velocity to zero.
- In 2D examples/tests this is given by `uγ = (zeros(nt), zeros(nt))`.

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
| Embedded interface BC | Embedded inflow/outflow scalar imposition | Not supported | Inflow/outflow BCs are only for outer box boundaries |

## Current Limitation

- Embedded inflow/outflow scalar boundary conditions on `Γ` are not supported; embedded-interface mode is no-flow (`u·n=0`).

## Public API

Main exported interface:
- `TransportModelMono`
- `assemble_steady_mono!`
- `assemble_unsteady_mono!`
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

## Build Docs

```bash
julia --project=docs docs/make.jl
```
