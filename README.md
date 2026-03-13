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

Fixed geometry uses `s = uγ·nγ`.  
Moving geometry uses relative normal speed `λ = (uγ - wγ)·nγ`.

Mono:

- fixed: if `s < 0` and interface inflow data are provided, impose inflow Dirichlet `Tγ = g`,
- moving: if `λ < 0` and interface inflow data are provided, impose inflow Dirichlet `Tγ = g`,
- otherwise use continuity closure `Tγ = Tω`.

Two-phase:

- closure is selected locally from phase-wise signs (`s1,s2` fixed; `λ1,λ2` moving),
- only inflow information is imposed,
- the ill-posed both-inflow local configuration is rejected with `ArgumentError`.

No-flow mode is recovered by setting interface velocity to zero (`uγ = 0`).

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

## Convergence Matrix (Tmp Environment)

Settings used for all rows:

- 1D periodic setup
- grids: `n = 33, 65, 129`
- `dt = 0.4 * h / U`, `U = 0.4`, `Tend = 0.1`
- error: weighted `L2` on active cells
- exact sampled at final `cap.C_ω`
- recomputed on `2026-03-13` with `/tmp/pt_convergence_matrix.jl` (post right-inflow outer-BC fix)

### Mono

| Geometry | Interface | Space | Time | e(33) | e(65) | e(129) | p33→65 | p65→129 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fixed | without | Upwind1 | BE | 2.352e-02 | 1.196e-02 | 6.056e-03 | 0.976 | 0.982 |
| fixed | without | Upwind1 | CN | 1.720e-02 | 8.665e-03 | 4.348e-03 | 0.989 | 0.995 |
| fixed | without | Centered | BE | 6.659e-03 | 3.358e-03 | 1.722e-03 | 0.988 | 0.964 |
| fixed | without | Centered | CN | 1.224e-03 | 3.069e-04 | 7.694e-05 | 1.996 | 1.996 |
| fixed | with | Upwind1 | BE | 1.018e+42 | 1.025e+01 | 5.191e+00 | 136.189 | 0.982 |
| fixed | with | Upwind1 | CN | 3.226e+00 | 2.896e+00 | 3.026e+00 | 0.156 | -0.063 |
| fixed | with | Centered | BE | 9.250e+41 | 9.375e+00 | 4.875e+00 | 136.180 | 0.943 |
| fixed | with | Centered | CN | 2.948e+00 | 2.649e+00 | 2.842e+00 | 0.155 | -0.102 |
| moving | without | Upwind1 | BE | 2.352e-02 | 1.196e-02 | 6.056e-03 | 0.976 | 0.982 |
| moving | without | Upwind1 | CN | 1.720e-02 | 8.665e-03 | 4.348e-03 | 0.989 | 0.995 |
| moving | without | Centered | BE | 6.659e-03 | 3.358e-03 | 1.722e-03 | 0.988 | 0.964 |
| moving | without | Centered | CN | 1.224e-03 | 3.069e-04 | 7.694e-05 | 1.996 | 1.996 |
| moving | with | Upwind1 | BE | 4.799e-02 | 2.816e-02 | 1.627e-02 | 0.769 | 0.792 |
| moving | with | Upwind1 | CN | 4.140e-02 | 2.437e-02 | 1.343e-02 | 0.764 | 0.860 |
| moving | with | Centered | BE | 2.916e-02 | 1.548e-02 | 8.271e-03 | 0.913 | 0.905 |
| moving | with | Centered | CN | 2.428e-02 | 1.332e-02 | 6.662e-03 | 0.866 | 0.999 |

### Two-Phase

| Geometry | Interface | Space | Time | e1(33) | e1(65) | e1(129) | p1 33→65 | p1 65→129 | e2(33) | e2(65) | e2(129) | p2 33→65 | p2 65→129 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | without | Upwind1 | BE | 2.352e-02 | 1.196e-02 | 6.056e-03 | 0.976 | 0.982 | 2.352e-02 | 1.196e-02 | 6.056e-03 | 0.976 | 0.982 |
| fixed | without | Upwind1 | CN | 1.720e-02 | 8.665e-03 | 4.348e-03 | 0.989 | 0.995 | 1.720e-02 | 8.665e-03 | 4.348e-03 | 0.989 | 0.995 |
| fixed | without | Centered | BE | 6.659e-03 | 3.358e-03 | 1.722e-03 | 0.988 | 0.964 | 6.659e-03 | 3.358e-03 | 1.722e-03 | 0.988 | 0.964 |
| fixed | without | Centered | CN | 1.224e-03 | 3.069e-04 | 7.694e-05 | 1.996 | 1.996 | 1.224e-03 | 3.069e-04 | 7.694e-05 | 1.996 | 1.996 |
| fixed | with | Upwind1 | BE | 1.937e+00 | 1.249e+00 | 1.046e+00 | 0.633 | 0.256 | 2.689e-01 | 2.494e-01 | 1.333e+00 | 0.109 | -2.418 |
| fixed | with | Upwind1 | CN | 8.571e-01 | 8.710e-01 | 8.767e-01 | -0.023 | -0.010 | 2.092e-01 | 2.180e-01 | 1.036e+00 | -0.059 | -2.248 |
| fixed | with | Centered | BE | 1.652e+00 | 1.092e+00 | 9.511e-01 | 0.597 | 0.200 | 2.584e-01 | 2.371e-01 | 1.245e+00 | 0.124 | -2.393 |
| fixed | with | Centered | CN | 7.402e-01 | 7.583e-01 | 7.965e-01 | -0.035 | -0.071 | 2.049e-01 | 2.068e-01 | 9.675e-01 | -0.013 | -2.226 |
| moving | without | Upwind1 | BE | 2.352e-02 | 1.196e-02 | 6.056e-03 | 0.976 | 0.982 | 2.352e-02 | 1.196e-02 | 6.056e-03 | 0.976 | 0.982 |
| moving | without | Upwind1 | CN | 1.720e-02 | 8.665e-03 | 4.348e-03 | 0.989 | 0.995 | 1.720e-02 | 8.665e-03 | 4.348e-03 | 0.989 | 0.995 |
| moving | without | Centered | BE | 6.659e-03 | 3.358e-03 | 1.722e-03 | 0.988 | 0.964 | 6.659e-03 | 3.358e-03 | 1.722e-03 | 0.988 | 0.964 |
| moving | without | Centered | CN | 1.224e-03 | 3.069e-04 | 7.694e-05 | 1.996 | 1.996 | 1.224e-03 | 3.069e-04 | 7.694e-05 | 1.996 | 1.996 |
| moving | with | Upwind1 | BE | 4.799e-02 | 2.816e-02 | 1.627e-02 | 0.769 | 0.792 | 3.214e-02 | 1.903e-02 | 1.100e-02 | 0.756 | 0.791 |
| moving | with | Upwind1 | CN | 4.140e-02 | 2.437e-02 | 1.343e-02 | 0.764 | 0.860 | 2.697e-02 | 1.564e-02 | 8.935e-03 | 0.786 | 0.808 |
| moving | with | Centered | BE | 2.916e-02 | 1.548e-02 | 8.271e-03 | 0.913 | 0.905 | 1.779e-02 | 9.742e-03 | 5.180e-03 | 0.869 | 0.911 |
| moving | with | Centered | CN | 2.428e-02 | 1.332e-02 | 6.662e-03 | 0.866 | 0.999 | 1.415e-02 | 7.693e-03 | 4.074e-03 | 0.879 | 0.917 |

## Local Docs Build

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
