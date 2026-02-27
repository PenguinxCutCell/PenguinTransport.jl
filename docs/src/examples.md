# Examples

Run from repository root:

```bash
julia --project examples/<file>.jl
```

## `advection_1d_periodic.jl`

- Periodic Gaussian advection.
- Compares `Upwind1` vs `MUSCL`.
- Uses `cfl_dt` and fixed-step Rosenbrock23.

## `advection_1d_sine_convergence.jl`

- Full-domain periodic advection (`levelset = -1`) with analytical sine-wave solution.
- Uses manual mass-matrix-aware SSP RK2 and `CFL = 0.5`.
- Reports refinement errors/orders in `L1` and `L2` for:
  - `Centered`
  - `MUSCL(MC())`
  - `MUSCL(VanLeer())`
- Uses `L1` as the primary limiter-scheme convergence metric.

## `advection_diffusion_1d.jl`

- Periodic smooth advection-diffusion with `kappa > 0`.
- Demonstrates CFL-based step selection with diffusion term included.

## `rotating_2d_transport.jl`

- 2D rotating velocity field.
- Demonstrates full-domain transport setup in two dimensions.

## `transport_diffusion_strang.jl`

- Coupling sketch using Strang splitting:
  - half diffusion step
  - full transport step
  - half diffusion step
- Uses two `TransportSystem`s with different physics settings.
