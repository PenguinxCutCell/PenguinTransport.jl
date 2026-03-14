# PenguinTransport.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/PenguinTransport.jl/dev/)
[![CI](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl)

`PenguinTransport.jl` is a cut-cell advection transport package for Cartesian grids in the PenguinxCutCell ecosystem (`CartesianGeometry.jl`, `CartesianOperators.jl`, `PenguinBCs.jl`, `PenguinSolverCore.jl`).

It supports:

- mono transport (steady and unsteady),
- two-phase transport (steady and unsteady),
- moving mono transport (unsteady),
- moving two-phase transport (unsteady),
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

All unsteady entry points (`assemble_unsteady_*`, `assemble_unsteady_*_moving!`, `solve_unsteady!`, `solve_unsteady_moving!`) accept exactly:

- `:BE` (Backward Euler, `θ = 1`)
- `:CN` (Crank-Nicolson, `θ = 1/2`)
- numeric `θ` with `0 <= θ <= 1`

Any unsupported symbol or numeric `θ` outside `[0,1]` raises `ArgumentError`.

## Embedded Interface Convention

Fixed geometry uses the discrete interface coefficient `κ` extracted from `ops.K` (cellwise embedded-interface transport term on ω rows).  
Moving geometry uses the discrete relative coefficient `κrel`, assembled from `(uγ - wγ)`.

Mono:

- if `κ < 0` (or `κrel < 0` moving) and interface inflow data are provided, impose inflow Dirichlet `Tγ = g`,
- otherwise use continuity closure `Tγ = Tω`.

Two-phase:

- closure is selected locally from phase-wise discrete signs (`κ1,κ2` fixed; `κ1rel,κ2rel` moving),
- donor side uses continuity closure, receiver side uses discrete flux continuity,
- the ill-posed both-inflow local configuration is rejected with `ArgumentError`.

No-flow mode is recovered by setting interface velocity to zero (`uγ = 0`) in fixed geometry, or `uγ = wγ` in moving geometry.

## Current Limitation (Two-Phase)

At a given interface location, if both phases are locally inflow simultaneously, the local closure is ill-posed for pure advection and assembly throws `ArgumentError` (fixed and moving models).

## Example Scripts

- `examples/smooth_blob_translation.jl`
- `examples/sharp_peak_advection.jl`
- `examples/manufactured_solution.jl`
- `examples/embedded_interface_bc_validation.jl`
- `examples/two_phase_planar_1d_validation.jl`
- `examples/two_phase_2d_planar_sanity.jl`
- `examples/moving_mono_material_translation.jl`
- `examples/moving_mono_interface_inflow.jl`
- `examples/moving_two_phase_planar_translation.jl`
- `examples/moving_two_phase_relative_flux_demo.jl`

## Convergence Matrix

### Mono

| Case | Space | Time | e(33) | e(65) | e(129) | p33→65 | p65→129 |
|---|---|---:|---:|---:|---:|---:|---:|
mono_fixed_no_interface|Upwind1|BE|2.352e-02|1.196e-02|6.056e-03|0.976|0.982
mono_fixed_no_interface|Upwind1|CN|1.720e-02|8.665e-03|4.348e-03|0.989|0.995
mono_fixed_no_interface|Centered|BE|6.659e-03|3.358e-03|1.722e-03|0.988|0.964
mono_fixed_no_interface|Centered|CN|1.224e-03|3.069e-04|7.694e-05|1.996|1.996
mono_fixed_interface|Upwind1|BE|3.846e-02|3.497e-02|3.392e-02|0.137|0.044
mono_fixed_interface|Upwind1|CN|4.018e-02|3.604e-02|3.452e-02|0.157|0.062
mono_fixed_interface|Centered|BE|1.603e-02|3.189e-02|6.382e+00|-0.992|-7.645
mono_fixed_interface|Centered|CN|1.538e-02|2.065e-02|4.509e-01|-0.425|-4.449
mono_moving_no_interface|Upwind1|BE|2.352e-02|1.196e-02|6.056e-03|0.976|0.982
mono_moving_no_interface|Upwind1|CN|1.720e-02|8.665e-03|4.348e-03|0.989|0.995
mono_moving_no_interface|Centered|BE|6.659e-03|3.358e-03|1.722e-03|0.988|0.964
mono_moving_no_interface|Centered|CN|1.224e-03|3.069e-04|7.694e-05|1.996|1.996
mono_moving_material_interface|Upwind1|BE|4.060e-02|2.209e-02|1.216e-02|0.878|0.862
mono_moving_material_interface|Upwind1|CN|3.419e-02|1.795e-02|9.754e-03|0.930|0.880
mono_moving_material_interface|Centered|BE|2.182e-02|1.093e-02|5.877e-03|0.998|0.895
mono_moving_material_interface|Centered|CN|1.679e-02|7.870e-03|4.383e-03|1.093|0.844

### Two-Phase

| Case | Space | Time | e1(33) | e1(65) | e1(129) | p1 33→65 | p1 65→129 | e2(33) | e2(65) | e2(129) | p2 33→65 | p2 65→129 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
two_fixed_no_interface|Upwind1|BE|2.352e-02|1.196e-02|6.056e-03|0.976|0.982|2.352e-02|1.196e-02|6.056e-03|0.976|0.982
two_fixed_no_interface|Upwind1|CN|1.720e-02|8.665e-03|4.348e-03|0.989|0.995|1.720e-02|8.665e-03|4.348e-03|0.989|0.995
two_fixed_no_interface|Centered|BE|6.659e-03|3.358e-03|1.722e-03|0.988|0.964|6.659e-03|3.358e-03|1.722e-03|0.988|0.964
two_fixed_no_interface|Centered|CN|1.224e-03|3.069e-04|7.694e-05|1.996|1.996|1.224e-03|3.069e-04|7.694e-05|1.996|1.996
two_fixed_interface_same_scalar|Upwind1|BE|2.626e-02|1.315e-02|5.250e-03|0.998|1.324|2.317e-02|1.191e-02|6.751e-03|0.960|0.819
two_fixed_interface_same_scalar|Upwind1|CN|4.349e-02|2.580e-02|1.249e-02|0.753|1.046|2.854e-02|1.403e-02|7.240e-03|1.024|0.955
two_fixed_interface_same_scalar|Centered|BE|6.737e-02|9.666e-02|1.097e-01|-0.521|-0.182|6.352e-02|9.111e-02|1.116e-01|-0.520|-0.293
two_fixed_interface_same_scalar|Centered|CN|7.339e-02|1.005e-01|1.145e-01|-0.454|-0.188|5.872e-02|9.215e-02|1.175e-01|-0.650|-0.350
two_moving_no_interface|Upwind1|BE|2.352e-02|1.196e-02|6.056e-03|0.976|0.982|2.352e-02|1.196e-02|6.056e-03|0.976|0.982
two_moving_no_interface|Upwind1|CN|1.720e-02|8.665e-03|4.348e-03|0.989|0.995|1.720e-02|8.665e-03|4.348e-03|0.989|0.995
two_moving_no_interface|Centered|BE|6.659e-03|3.358e-03|1.722e-03|0.988|0.964|6.659e-03|3.358e-03|1.722e-03|0.988|0.964
two_moving_no_interface|Centered|CN|1.224e-03|3.069e-04|7.694e-05|1.996|1.996|1.224e-03|3.069e-04|7.694e-05|1.996|1.996
two_moving_material_interface_same_scalar|Upwind1|BE|4.060e-02|2.209e-02|1.216e-02|0.878|0.862|2.262e-02|1.306e-02|7.602e-03|0.792|0.781
two_moving_material_interface_same_scalar|Upwind1|CN|3.419e-02|1.795e-02|9.754e-03|0.930|0.880|1.793e-02|1.029e-02|5.947e-03|0.801|0.791
two_moving_material_interface_same_scalar|Centered|BE|2.182e-02|1.093e-02|5.877e-03|0.998|0.895|1.023e-02|6.178e-03|3.699e-03|0.728|0.740
two_moving_material_interface_same_scalar|Centered|CN|1.679e-02|7.870e-03|4.383e-03|1.093|0.844|8.282e-03|4.701e-03|2.763e-03|0.817|0.767
two_moving_material_interface_same_scalar|Centered|CN|1.679e-02|7.870e-03|4.383e-03|1.093|0.844|8.282e-03|4.701e-03|2.763e-03|0.817|0.767

## Local Docs Build

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
