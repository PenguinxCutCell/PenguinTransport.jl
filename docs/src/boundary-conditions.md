# Boundary Conditions

Two BC sets are used:

- Diffusion BC: `bc_diff::CartesianOperators.BoxBC`
- Advection BC: `bc_adv::CartesianOperators.AdvBoxBC`

## Diffusion BC (`BoxBC`)

Supported by `CartesianOperators.kernel_ops(...; bc=bc_diff)`:

- `Dirichlet`
- `Neumann`
- `Periodic`

Default (`bc_diff=nothing`) is all-zero Neumann via `BoxBC(Val(N), T)`.

## Advection BC (`AdvBoxBC`)

Supported by kernel convection:

- `AdvPeriodic`
- `AdvOutflow`
- `AdvInflow(value)`

Default (`bc_adv=nothing`) is all outflow via `AdvBoxBC(Val(N), T)`.

## Stencil Behavior

Advection internally uses a stencil BC derived from advection periodicity:

- periodic advection side -> periodic stencil.
- non-periodic advection side -> zero-Neumann stencil.

The advection BC object is still passed to convection kernels to apply inflow/outflow logic.

## Runtime BC Updates

`AdvBCUpdater` can update advection BC during integration.

- If periodicity pattern changes, system marks operators dirty and triggers rebuild.
- If periodicity is unchanged, update is `:rhs_only`.

## Embedded Interface Inflow (Cut Cells)

`TransportProblem` supports `embedded_inflow=...` for cut-cell advection.

- This is an interface trace closure on the embedded boundary `Γ`, not a box-boundary condition.
- The prescribed trace is imposed only on inflow interface segments (`u⋅n < 0`).
- On outflow (`u⋅n >= 0`), no value is imposed and the interior trace is used.

Accepted payloads:

- scalar (`embedded_inflow=1.0`)
- full `Nd` vector
- callback returning scalar/vector, or callback that fills destination buffer in-place
