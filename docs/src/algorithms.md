# Algorithms

## 1. Discrete Unknown Layouts

Mono:

```text
x = [ φω ; φγ ]
```

Two-phase:

```text
x = [ φω1 ; φγ1 ; φω2 ; φγ2 ]
```

The two-phase ordering is fixed in all public APIs.

## 2. Core Advection Blocks

`CartesianOperators.jl` provides cut-cell advection operators.

- `ω` rows receive bulk transport + source + (unsteady) time terms
- `γ` rows enforce local embedded-interface closure

For moving models, interface transport/closure uses the discrete relative interface coefficient assembled with `(uγ-wγ)`.

## 3. Steady Mono Assembly

Block form:

```math
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
\phi_\omega\\
\phi_\gamma
\end{bmatrix}
=
\begin{bmatrix}
b_\omega\\
b_\gamma
\end{bmatrix}.
```

- `A11`, `A12`: advection blocks
- `A21`, `A22`: interface closure rows (`Tγ=g` inflow or `Tγ=Tω` continuity)
- outer box BCs are then applied on `ω` rows

## 4. Unsteady Mono `θ` Assembly (Fixed)

Fixed geometry uses the usual `θ` method on `ω` rows:

```math
\left(\frac{V}{\Delta t} + \theta A\right)\phi^{n+1}
=
\frac{V}{\Delta t}\phi^n - (1-\theta)A\phi^n + b.
```

`solve_unsteady!` reuses constant operators when matrix/RHS are time-invariant.

## 5. Unsteady Mono Moving-Slab Assembly

Moving geometry uses reduced slab volumes `Vn`, `Vn1`:

```math
A_{11} = M_1 + A^{adv}_{11}\Psi_+, \qquad
A_{12} = -(M_1-M_0) + A^{adv}_{12}\Psi_+,
```

```math
b_\omega = (M_0 - A^{adv}_{11}\Psi_-)\phi_\omega^n
          - (A^{adv}_{12}\Psi_-)\phi_\gamma^n
          + b_{src}.
```

with `M0=diag(Vn)`, `M1=diag(Vn1)`.

The `-(M1-M0)` term carries the pure geometry sweep effect, so no extra standalone geometric flux term is added.

## 6. Steady/Unsteady Two-Phase (Fixed)

Per-phase `ω` blocks are assembled like mono.

`γ1`/`γ2` rows are selected locally from discrete sign logic (`κ1`, `κ2`):

- one inflow / one outflow: coupling row + outflow continuity row
- both outflow: continuity on both
- both inflow: rejected (ill-posed local closure)

## 7. Unsteady Two-Phase Moving-Slab

Same block philosophy as fixed two-phase, but with:

- per-phase moving time terms from `V1n/V1n1`, `V2n/V2n1`
- interface sign logic based on discrete relative coefficients `κ1rel`, `κ2rel`
- both-inflow rejection based on those discrete signs

Outer box BC logic remains driven by bulk velocities `u1ω`, `u2ω` (not `wγ`).

## 8. Regularization / Inactive Rows

Inactive and halo rows are forced to identity equations.

This keeps matrix layouts stable and avoids singular systems as active topology changes.

## 9. Practical Notes

- `Centered()` is less diffusive but can oscillate near sharp fronts.
- `Upwind1()` is more robust/monotone but diffusive.
- For moving models, expect first-order-in-time behavior with `:BE`; `:CN` gives less temporal damping but follows the same validated `θ` contract.
