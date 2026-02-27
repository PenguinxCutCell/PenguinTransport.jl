# Design Notes

## Reduced-State Architecture

- The solver state is reduced to active `\omega` cells only.
- Full vectors (`Tω_full`, `Tγ_full`, `du_full`) are internal caches.
- Restriction/prolongation use `PenguinSolverCore.DofMap`.

## `\gamma` Handling

- Current default is monophasic closure:
  - `Tγ_full := Tω_full`
- Optional internal `gammafun` hook allows override payloads:
  - scalar
  - full-length vector
  - reduced-length vector mapped to active `\omega`.

## Velocity Payload Normalization

`vel_omega` / `vel_gamma` accept:

- scalar (1D only)
- tuple of scalars (`NTuple{N,Number}`)
- tuple of full/reduced vectors
- callables with polymorphic signatures:
  - `(sys, u, p, t)`
  - `(u, p, t)`
  - `(t)`
  - `()`

## Rebuild Contract

- `rebuild!` refreshes kernel operators only when `ops_dirty=true`.
- `AdvBCUpdater` triggers `:matrix` rebuild only when periodicity pattern changes.

## Steady Matrix-Free Solve

- `steady_linear_problem` freezes runtime fields at `(u_eval, p, t)`.
- Builds a `FunctionOperator` evaluating `A*x` from kernel transport on full caches.
- Affine transport offset and source are moved to RHS.
