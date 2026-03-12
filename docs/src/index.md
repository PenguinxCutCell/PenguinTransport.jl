# PenguinTransport.jl

`PenguinTransport.jl` solves cut-cell advection transport problems on Cartesian grids with fixed and moving embedded interfaces. It provides mono and two-phase models, with fixed-geometry steady/unsteady assembly and moving-geometry unsteady slab assembly.

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
| Models | Moving monophase unsteady | Implemented | `MovingTransportModelMono` + `assemble_unsteady_mono_moving!` |
| Models | Moving twophase unsteady | Implemented | `MovingTransportModelTwoPhase` + `assemble_unsteady_two_phase_moving!` |
| Advection | Centered scheme | Implemented | `Centered()` spatial discretization |
| Advection | Upwind scheme | Implemented | `Upwind1()` first-order upwind |
| Time schemes | Backward Euler | Implemented | `theta=1` |
| Time schemes | Crank–Nicolson | Implemented | `theta=0.5` |
| Time schemes | General theta | Implemented | Numeric `theta` with `0 ≤ theta ≤ 1` |
| Outer BCs | Inflow | Implemented | Dirichlet-type boundary condition |
| Outer BCs | Outflow | Implemented | Neumann-type boundary condition |
| Outer BCs | Periodic | Implemented | Periodic boundary condition |
| Embedded interface | Sign-based closure | Implemented | Fixed: `s = uγ·nγ`; Moving: `λ = (uγ-wγ)·nγ` |


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

Moving quick-start APIs are `MovingTransportModelMono`, `MovingTransportModelTwoPhase`, and `solve_unsteady_moving!` (see [API](api.md) and [Examples](examples.md)).

## Local Docs Build

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
