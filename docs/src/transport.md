# Transport Models

## 1. Monophasic Scalar Transport

Fixed geometry (`Ω`, `Γ` fixed):

```math
\partial_t \phi + \nabla\cdot(\mathbf{u}_\omega\,\phi) = s,
```

with steady limit

```math
\nabla\cdot(\mathbf{u}_\omega\,\phi) = s.
```

Moving geometry (`Ω(t)`, `Γ(t)`):

```math
\partial_t \phi + \nabla\cdot(\mathbf{u}_\omega\,\phi) = s \quad \text{in } \Omega(t),
```

with interface velocity `wγ`; moving embedded-interface transport is assembled with relative interface velocity `(uγ - wγ)`, producing the discrete coefficient `κrel` used by closure logic.

`uω` is bulk velocity, `uγ` is interface-sampled advective velocity, and `φγ` are interface unknowns used for local closure rows.

## 2. Two-Phase Scalar Transport

Fixed geometry:

```math
\partial_t \phi_1 + \nabla\cdot(\mathbf{u}_{\omega,1}\,\phi_1) = s_1,
```

```math
\partial_t \phi_2 + \nabla\cdot(\mathbf{u}_{\omega,2}\,\phi_2) = s_2.
```

Moving geometry uses `Ω1(t), Ω2(t)` and per-phase relative interface velocities `(uγ,1-wγ)` and `(uγ,2-wγ)` to assemble discrete coefficients `κ1rel`, `κ2rel` for closure switching.

Unknown ordering is always:

```text
(ω1, γ1, ω2, γ2)
```

This is a hyperbolic inflow/outflow closure problem, not a diffusion-style double-jump constraint.

## 3. Outer Boundary Conditions

Supported advection BCs:

- `Inflow(value)`
- `Outflow()`
- `Periodic()`

Only inflow boundaries require imposed scalar values.

## 4. Embedded-Interface Convention

The implementation does not branch from a pointwise `u·n` probe.
Inflow/outflow switching is done from the same discrete embedded-interface coefficient used by the `ω` rows:

```math
\kappa_i = \sum_d \mathrm{diag}(K_d)_i.
```

### Fixed geometry sign gate

```math
\text{fixed gate}:\quad \kappa_i < 0.
```

### Moving geometry sign gate

```math
\text{moving gate}:\quad \kappa^{rel}_i < 0,
\quad \kappa^{rel} \text{ assembled from } (u_\gamma-w_\gamma).
```

### Mono closure

- fixed: if `κ < 0` and interface data exist, impose inflow Dirichlet `Tγ = g`
- moving: if `κrel < 0` and interface data exist, impose inflow Dirichlet `Tγ = g`
- otherwise use continuity closure `Tγ = Tω`

`|κ|` / `|κrel|` near machine zero is treated as non-inflow (continuity/outflow behavior).

### Two-phase closure

Fixed uses `(κ1, κ2)`, moving uses `(κ1rel, κ2rel)` with the same local logic:

- one inflow / one outflow: conservative transport coupling row on inflow side, continuity closure on outflow side
- both outflow: continuity closure on both phases
- both inflow at the same interface location: rejected as ill-posed (`ArgumentError`)

## 5. Moving-Slab Conservative Interpretation

Moving models are assembled from reduced space-time slabs with physical volumes `Vn` and `Vn1`.

- time terms use `M0 = diag(Vn)` and `M1 = diag(Vn1)`
- geometric sweep is represented by `Vn1 - Vn`
- no standalone extra geometric flux term is added

This is why moving interface logic uses the relative discrete coefficient in closure and interface advection treatment.

## 6. Callback Conventions

Accepted scalar/velocity inputs:

- constants
- space callbacks `(x...)`
- time callbacks `(x..., t)`

Velocity input layout:

- mono: `uω`, `uγ`, and for moving also `wγ`
- two-phase: `u1ω`, `u1γ`, `u2ω`, `u2γ`, and for moving `wγ`
- each component can be a vector of length `ntotal`, constant, or callback

## 7. Scheme Conventions

All unsteady APIs (fixed and moving) accept:

- `:BE` (`θ = 1`)
- `:CN` (`θ = 1/2`)
- numeric `θ` with `0 <= θ <= 1`

Anything else throws `ArgumentError`.
