# PenguinTransport.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/)
[![CI](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl)

`PenguinTransport.jl` is a cut-cell advection transport package for Cartesian grids in the PenguinxCutCell ecosystem (`CartesianGeometry.jl`, `CartesianOperators.jl`, `PenguinBCs.jl`, `PenguinSolverCore.jl`).

It supports:

- mono transport (steady and unsteady),
- two-phase transport (steady and unsteady),
- spatial advection schemes `Centered()` and `Upwind1()`,
- outer advection BCs `Inflow` / `Outflow` / `Periodic`,
- embedded-interface sign-based closure with unknown ordering `(ω1, γ1, ω2, γ2)` for two-phase.

## Governing Equation

```math
\partial_t \phi + \nabla\cdot(\mathbf{u}\,\phi) = s
```

Steady form:

```math
\nabla\cdot(\mathbf{u}\,\phi) = s
```

## Documentation

- Home: <https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/>
- API: <https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/api/>
- Examples: <https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/examples/>
- Algorithms: <https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/algorithms/>
- Transport Models: <https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/transport/>

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/PenguinxCutCell/PenguinTransport.jl")
```

## Quick Start (Mono Unsteady)

```julia
using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinTransport

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

grid = (0.0:0.05:1.0,)
cap = assembled_capacity(full_moments(grid); bc=0.0)
nt = cap.ntotal

bc = BorderConditions(; left=Periodic(), right=Periodic())
uω = (ones(nt),)
uγ = (ones(nt),)

model = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Centered())
res = solve_unsteady!(model, zeros(nt), (0.0, 0.2); dt=0.01, scheme=:CN)
```

## Time Schemes

All unsteady entry points (`assemble_unsteady_*`, `solve_unsteady!`) accept exactly:

- `:BE` (Backward Euler, `θ = 1`)
- `:CN` (Crank-Nicolson, `θ = 1/2`)
- numeric `θ` with `0 <= θ <= 1`

Any unsupported symbol or numeric `θ` outside `[0,1]` raises `ArgumentError`.

## Embedded Interface Convention

Let `s = uγ·nγ`.

Mono:

- if `s < 0` and interface inflow data are provided, impose inflow Dirichlet `Tγ = g`,
- otherwise use continuity closure `Tγ = Tω`.

Two-phase:

- closure is selected locally from phase-wise signs,
- only inflow information is imposed,
- the ill-posed both-inflow local configuration is rejected with `ArgumentError`.

No-flow mode is recovered by setting interface velocity to zero (`uγ = 0`).

## Example Scripts

- `examples/smooth_blob_translation.jl`
- `examples/sharp_peak_advection.jl`
- `examples/manufactured_solution.jl`
- `examples/embedded_interface_bc_validation.jl`
- `examples/two_phase_planar_1d_validation.jl`
- `examples/two_phase_2d_planar_sanity.jl`

## Local Docs Build

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
