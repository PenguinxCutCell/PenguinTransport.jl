# PenguinTransport.jl

[![CI](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/PenguinxCutCell/PenguinTransport.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PenguinxCutCell/PenguinTransport.jl)

`PenguinTransport.jl` is a cut-cell scalar transport package built on `CartesianGeometry.jl`, `CartesianOperators.jl`, and `PenguinSolverCore.jl`.

It solves:

```math
M \dot T = \mathcal C(u^\omega, u^\gamma;\text{scheme})\,T + \kappa\,\Delta(T^\omega, T^\gamma) + S
```

with reduced-state unknowns on active `\omega` cells and kernel-based operators.

## Features

- Reduced state on active `\omega` cells (`DofMap`) with diagonal mass matrix.
- Kernel-based advection (`Centered`, `Upwind1`, `MUSCL`) and optional diffusion.
- Runtime updaters (`kappa`, scheme, velocity, advection BC, source) via `PenguinSolverCore.UpdateManager`.
- Matrix-free steady solve API (`steady_linear_problem`, `steady_solve`) via `LinearSolve.FunctionOperator`.
- CFL helper (`cfl_dt`) for advection and advection-diffusion setups.
- Validation tests for conservation, accuracy order, boundedness, and manufactured advection-diffusion.

## Quickstart

```julia
using PenguinTransport, CartesianGeometry, CartesianOperators, PenguinSolverCore
x = collect(range(0.0, 1.0; length=129)); full(_x, _=0.0) = -1.0
mom = CartesianGeometry.geometric_moments(full, (x,), Float64, zero; method=:implicitintegration)
bc_adv = CartesianOperators.AdvBoxBC((CartesianOperators.AdvPeriodic(Float64),), (CartesianOperators.AdvPeriodic(Float64),))
prob = TransportProblem(; kappa=0.01, bc_adv=bc_adv, scheme=CartesianOperators.Upwind1(), vel_omega=1.0, vel_gamma=1.0)
sys = build_system(mom, prob)
u0 = rand(length(sys.dof_omega.indices)); du = similar(u0)
dt = cfl_dt(sys, u0; cfl=0.45)
PenguinSolverCore.rhs!(du, sys, u0, nothing, 0.0)
```

## Supported BCs and Schemes

- Diffusion BC: `CartesianOperators.BoxBC` (`Dirichlet`, `Neumann`, `Periodic`) through kernel diffusion ops.
- Advection BC: `CartesianOperators.AdvBoxBC` (`AdvPeriodic`, `AdvOutflow`, `AdvInflow`).
- Advection schemes: `Centered()`, `Upwind1()`, `MUSCL(limiter)` (`Minmod`, `MC`, `VanLeer`).

## Example Snapshot

From `examples/advection_1d_periodic.jl` (Upwind1 vs MUSCL at `t=0.3`):

![Periodic 1D advection comparison](docs/assets/advection_1d_snapshot.svg)

## Design Notes

- State vector is **omega-reduced only** (`u::Vector` over active `\omega` DOFs).
- Full fields (`Tω_full`, `Tγ_full`) are internal caches used for kernel evaluation.
- Default monophasic closure is `Tγ_full = Tω_full`; a `gammafun` hook is available internally for overrides.
- Source callbacks return physical source density and are mass-weighted by cell volume in `rhs!`.
- Steady matrix-free solver freezes runtime coefficients at `(u_eval, p, t)` and solves `A(u) = b` using `LinearSolve`.

## Validation Contract (tests)

- Periodic advection mass conservation.
- Smooth advection convergence vs analytical sine-wave solution on `full_domain = -1`.
  - `Centered + RK2`: clear second-order in `L1`/`L2`.
  - `MUSCL(MC/VanLeer) + RK2`: clear second-order in `L1` (and convergent `L2` behavior).
- Nonsmooth boundedness / overshoot checks for limiter schemes.
- Manufactured advection-diffusion regression.
- `rhs!` allocation regression (performance gate).

## Examples

- `examples/advection_1d_periodic.jl`
- `examples/advection_1d_sine_convergence.jl` (analytical sine-wave order test, RK2, CFL=0.5)
- `examples/advection_diffusion_1d.jl`
- `examples/rotating_2d_transport.jl`
- `examples/transport_diffusion_strang.jl` (Strang splitting sketch)
