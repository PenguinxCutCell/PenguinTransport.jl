# PenguinTransport.jl

`PenguinTransport.jl` solves cut-cell advection transport problems on Cartesian grids with fixed embedded interfaces. It provides mono and two-phase models in steady and unsteady form, with assembly and solve wrappers designed to stay close to the underlying block system.

The conservative transport equation is:

```math
\partial_t \phi + \nabla\cdot(\mathbf{u}\,\phi) = s,
```

with steady limit 
```math
\nabla\cdot(\mathbf{u}\,\phi) = s
```

Within PenguinxCutCell, `PenguinTransport.jl` sits on top of:

- `CartesianGeometry.jl` for cut-cell moments,
- `CartesianOperators.jl` for advection operators,
- `PenguinBCs.jl` for boundary-condition types,
- `PenguinSolverCore.jl` for linear solves.

## Implemented Today

| Area | Item | Status | Notes |
|---|---|---|---|
| Models | Monophase steady | Implemented | `TransportModelMono` + `assemble_steady_mono!` |
| Models | Monophase unsteady | Implemented | Theta-form assembly via `assemble_unsteady_mono!` |
| Models | Twophase steady | Implemented | `TransportModelTwoPhase` + `assemble_steady_two_phase!` |
| Models | Twophase unsteady | Implemented | Theta-form assembly via `assemble_unsteady_two_phase!` |
| Advection | Centered scheme | Implemented | `Centered()` spatial discretization |
| Advection | Upwind scheme | Implemented | `Upwind1()` first-order upwind |
| Time schemes | Backward Euler | Implemented | `theta=1` |
| Time schemes | Crank–Nicolson | Implemented | `theta=0.5` |
| Time schemes | General theta | Implemented | Numeric `theta` with `0 ≤ theta ≤ 1` |
| Outer BCs | Inflow | Implemented | Dirichlet-type boundary condition |
| Outer BCs | Outflow | Implemented | Neumann-type boundary condition |
| Outer BCs | Periodic | Implemented | Periodic boundary condition |
| Embedded interface | Sign-based closure | Implemented | From `s = uγ·nγ` (inflow Dirichlet or continuity) |


## Pages

- [Transport Models](transport.md): PDEs, sign conventions, callback layout, time-scheme contract.
- [Algorithms](algorithms.md): unknown layouts, block assembly, theta-step construction, row regularization.
- [API](api.md): public constructors/functions and state-layout contracts.
- [Examples](examples.md): curated scripts and verification map.

## Quick Start

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

## Local Docs Build

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
