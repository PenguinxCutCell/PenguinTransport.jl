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

with interface velocity `wγ` and relative normal speed

```math
\lambda = (\mathbf{u}_\gamma - \mathbf{w}_\gamma)\cdot \mathbf{n}_\gamma.
```

`uω` is bulk velocity, `uγ` is interface-sampled advective velocity, and `φγ` are interface unknowns used for local closure rows.

## 2. Two-Phase Scalar Transport

Fixed geometry:

```math
\partial_t \phi_1 + \nabla\cdot(\mathbf{u}_{\omega,1}\,\phi_1) = s_1,
```

```math
\partial_t \phi_2 + \nabla\cdot(\mathbf{u}_{\omega,2}\,\phi_2) = s_2.
```

Moving geometry uses `Ω1(t), Ω2(t)` and relative interface speeds:

```math
\lambda_1 = (\mathbf{u}_{\gamma,1} - \mathbf{w}_\gamma)\cdot \mathbf{n}_{\gamma,1}, \qquad
\lambda_2 = (\mathbf{u}_{\gamma,2} - \mathbf{w}_\gamma)\cdot \mathbf{n}_{\gamma,2}.
```

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

### Fixed geometry sign

```math
s = u_\gamma \cdot n_\gamma.
```

### Moving geometry sign

```math
\lambda = (u_\gamma - w_\gamma)\cdot n_\gamma.
```

### Mono closure

- fixed: if `s < 0` and interface data exist, impose inflow Dirichlet `Tγ = g`
- moving: if `λ < 0` and interface data exist, impose inflow Dirichlet `Tγ = g`
- otherwise use continuity closure `Tγ = Tω`

`|λ|` near machine zero is treated as non-inflow (continuity/outflow behavior).

### Two-phase closure

Fixed uses `(s1, s2)`, moving uses `(λ1, λ2)` with the same local logic:

- one inflow / one outflow: conservative transport coupling row on inflow side, continuity closure on outflow side
- both outflow: continuity closure on both phases
- both inflow at the same interface location: rejected as ill-posed (`ArgumentError`)

## 5. Moving-Slab Conservative Interpretation

Moving models are assembled from reduced space-time slabs with physical volumes `Vn` and `Vn1`.

- time terms use `M0 = diag(Vn)` and `M1 = diag(Vn1)`
- geometric sweep is represented by `Vn1 - Vn`
- no standalone extra geometric flux term is added

This is why moving interface logic uses relative speed in closure and interface advection treatment.

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
