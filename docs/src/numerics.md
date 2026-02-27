# Numerics

## Active-Cell Masking

System build uses:

1. material mask: `cell_type != 0` and `V > vtol`
2. padding removal on high-index boundary plane (`padded_mask`)
3. active set `dof_omega = DofMap(findall(mask))`

Mass matrix:

```math
M = \mathrm{diag}(V_i)_{i \in \omega}
```

## Advection Schemes

`scheme::CartesianOperators.AdvectionScheme`:

- `Centered()`
- `Upwind1()`
- `MUSCL(limiter)` where limiter in `{Minmod, MC, VanLeer}`

## Kernel Work Buffers

RHS is evaluated with preallocated `KernelWork` buffers:

- `work_adv` for convection
- `work_diff` for diffusion/Laplacian

This keeps runtime allocations low.

## CFL Guidance

Use:

```julia
dt = cfl_dt(sys, u; cfl=0.45)
```

`cfl_dt` computes:

- advection restriction from `min(dx_d / max|u_d|)`
- optional diffusion restriction from `1 / (2*kappa*sum(1/dx_d^2))`

and returns `cfl * min(dt_adv, dt_diff)`.
