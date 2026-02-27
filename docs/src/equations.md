# Equations

`PenguinTransport.jl` solves scalar transport in semidiscrete form:

```math
M \dot T =
\mathcal C(u^\omega, u^\gamma; \text{scheme})\,T
+ \kappa\,\Delta(T^\omega, T^\gamma)
+ S
```

where:

- `M` is diagonal mass matrix built from active-cell volumes.
- `\mathcal C` is the kernel convection operator from `CartesianOperators`.
- `\Delta` is the kernel cut-cell Laplacian.
- `S` is source term (mass-weighted internally by cell volumes).

## State Representation

- Unknown state `u` is reduced to active `\omega` DOFs.
- Full fields are rebuilt internally each RHS call:
  - `Tω_full` via prolongation from reduced state.
  - `Tγ_full` defaults to `Tω_full` (monophasic closure), or optional override through `gammafun`.

## Special Cases

- Pure advection: `kappa == 0` uses `convection!`.
- Advection-diffusion: `kappa > 0` uses `advection_diffusion!`.

## Steady Matrix-Free Form

For steady solves, the package constructs a matrix-free linear problem:

```math
A x = b
```

with:

- `A` represented by `LinearSolve.FunctionOperator`.
- `A*x` evaluated by kernel transport operator on full caches, then restricted.
- affine part removed with a zero-state probe.
- source contribution shifted to the right-hand side.
