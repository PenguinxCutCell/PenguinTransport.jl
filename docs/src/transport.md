# Transport Models

## 1. Monophasic Scalar Transport

On a fixed cut-cell domain `Ω` with embedded interface `Γ`, the mono model solves:

```math
\partial_t \phi + \nabla\cdot(\mathbf{u}_\omega\,\phi) = s \quad \text{in } \Omega,
```

with steady form:

```math
\nabla\cdot(\mathbf{u}_\omega\,\phi) = s.
```

Here:

- `uω` is the bulk velocity field sampled on `ω` control volumes,
- `s` is the scalar source term,
- interface unknowns `φγ` are additional trace unknowns used to close embedded-interface rows.

## 2. Two-Phase Scalar Transport

For two phases separated by a fixed embedded interface `Γ`, the model solves one advection equation per phase:

```math
\partial_t \phi_1 + \nabla\cdot(\mathbf{u}_{\omega,1}\,\phi_1) = s_1,
```

```math
\partial_t \phi_2 + \nabla\cdot(\mathbf{u}_{\omega,2}\,\phi_2) = s_2.
```

Unknown ordering is fixed as:

```text
(ω1, γ1, ω2, γ2)
```

The interface closure is advection-driven: this is a first-order hyperbolic inflow/outflow closure problem, not a diffusion-style double-jump constraint. Locally, only inflow information is required; outflow states are propagated by continuity/transport closure.

## 3. Outer Boundary Conditions

`BorderConditions` supports advection boundary types:

- `Inflow(value)`
- `Outflow()`
- `Periodic()`

For pure advection, only inflow boundaries require imposed scalar values. Outflow boundaries do not require scalar data.

## 4. Embedded-Interface Convention

Define the local interface sign:

```math
s = u_\gamma \cdot n_\gamma.
```

### Mono case

- If `s < 0` and `bc_interface` provides a value, the interface row is inflow Dirichlet: `Tγ = g`.
- Otherwise (`s >= 0`, or no inflow value provided), the interface row uses continuity closure: `Tγ = Tω`.

### Two-phase case

The code evaluates `s1 = uγ1·nγ1` and `s2 = uγ2·nγ2` cellwise and selects a locally well-posed closure:

- one phase inflow / the other outflow: flux coupling row on the inflow side, continuity closure on the outflow side,
- both outflow: continuity closure on each phase,
- both inflow at the same interface location: rejected with `ArgumentError` as an ill-posed both-inflow local configuration.

### No-flow case

If `uγ = 0` (mono) or phase interface velocities are zero, no advective interface flux is present and the closure falls back to continuity rows.

### Sign convention 

The assembled advection operator follows a fixed internal sign convention. In particular, imposed advection inflow data enters with that operator sign; tests and examples lock this behavior explicitly so users can match expectations when building manufactured solutions.

## 5. Callback Conventions

Accepted scalar inputs (source, interface inflow values, velocity components):

- constants,
- space callbacks `(x...)`,
- time callbacks `(x..., t)` where applicable.

Velocity layout:

- Mono bulk/interface velocity inputs: `uω`, `uγ` are tuples (or vectors) of `N` components.
- Two-phase velocity inputs: `u1ω`, `u1γ`, `u2ω`, `u2γ` with the same component convention.
- Each component can be a full vector of length `ntotal`, a constant, or a callback.

## 6. Scheme Conventions

All unsteady APIs accept exactly:

- `:BE` for Backward Euler (`θ = 1`),
- `:CN` for Crank-Nicolson (`θ = 1/2`),
- numeric `θ` with `0 <= θ <= 1`.

Numeric values outside `[0,1]` are rejected with `ArgumentError`.
