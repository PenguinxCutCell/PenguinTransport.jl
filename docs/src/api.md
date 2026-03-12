# API

## Public Models

### `TransportModelMono(cap, ops, uω, uγ; kwargs...)`
### `TransportModelMono(cap, uω, uγ; kwargs...)`

Required inputs:

- `cap::AssembledCapacity`
- either prebuilt `ops::AdvectionOps` or velocities to build them
- `uω`, `uγ`: velocity containers with `N` components (tuple/vector), where each component is constant, callback, or vector of length `ntotal`

Key keywords:

- `source`: constant or callback `(x...)` / `(x..., t)`
- `bc_border::BorderConditions`
- `bc_interface`: `nothing`, `Inflow(value)`, `Dirichlet(value)`, scalar, or callback
- `layout`: default canonical mono layout (`ω`, then `γ`)
- `scheme`: spatial advection discretization (`Centered()` or `Upwind1()`)

Unknown layout:

```text
x = [ φω ; φγ ]
```

### `TransportModelTwoPhase(cap1, cap2, ops1, ops2, u1ω, u1γ, u2ω, u2γ; kwargs...)`
### `TransportModelTwoPhase(cap1, cap2, u1ω, u1γ, u2ω, u2γ; kwargs...)`

Required inputs:

- `cap1`, `cap2` (same `ntotal` and grid shape)
- per-phase velocities `u1ω`, `u1γ`, `u2ω`, `u2γ` with the same component conventions as mono

Key keywords:

- `source1`, `source2`
- `bc_border1`, `bc_border2`
- `layout`: default canonical two-phase layout
- `scheme`: spatial advection discretization (`Centered()` or `Upwind1()`)

Unknown layout (fixed):

```text
x = [ φω1 ; φγ1 ; φω2 ; φγ2 ]
```

## Assembly Functions

### `assemble_steady_mono!(sys, model, t)`

Mutates `sys.A`, `sys.b` for mono steady transport at time `t`.

### `assemble_unsteady_mono!(sys, model, uⁿ, t, dt, scheme_or_theta)`

Mutates `sys.A`, `sys.b` for one mono unsteady `θ` step.

- `uⁿ` can be `ω`-only (`ntotal`) or full state length.
- Accepted time schemes: `:BE`, `:CN`, numeric `θ` in `[0,1]`.

### `assemble_steady_two_phase!(sys, model, t)`

Mutates `sys.A`, `sys.b` for two-phase steady transport.

### `assemble_unsteady_two_phase!(sys, model, uⁿ, t, dt, scheme_or_theta)`

Mutates `sys.A`, `sys.b` for one two-phase unsteady `θ` step.

- `uⁿ` can be full state, concatenated `ω` blocks, or tuple `(u01, u02)`.
- Accepted time schemes: `:BE`, `:CN`, numeric `θ` in `[0,1]`.

## Solver Wrappers

### `solve_steady!(model; t=0, method=:direct, kwargs...)`

Assembles and solves one steady system. Returns `LinearSystem` with solution in `sys.x`.

### `solve_unsteady!(model, u0, tspan; dt, scheme=:BE, method=:direct, save_history=true, kwargs...)`

Time-integrates mono or two-phase transport and returns:

```text
(times, states, system, reused_constant_operator)
```

Important behavior:

- Accepted time schemes are exactly `:BE`, `:CN`, or numeric `θ ∈ [0,1]`.
- Invalid symbols or out-of-range `θ` raise `ArgumentError`.
- Reuses constant operator/factorization when matrix/RHS are time-invariant.

## Operator/Geometry Maintenance

### `update_advection_ops!(model; t=0)`

Rebuilds and stores advection operators at time `t`.

### `rebuild!(model, moments; bc=0, t=0)`

Rebuilds mono geometry from new moments and refreshes advection operators.

## Two-Phase Block Views

These helpers read sub-blocks from full two-phase state vectors:

- `omega1_view(model, x)`
- `gamma1_view(model, x)`
- `omega2_view(model, x)`
- `gamma2_view(model, x)`

All assume ordering `(ω1, γ1, ω2, γ2)`.

## Public Docstrings

```@docs
TransportModelMono
TransportModelTwoPhase
assemble_steady_mono!
assemble_unsteady_mono!
assemble_steady_two_phase!
assemble_unsteady_two_phase!
solve_steady!
solve_unsteady!
update_advection_ops!
rebuild!
omega1_view
gamma1_view
omega2_view
gamma2_view
```
