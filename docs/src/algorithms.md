# Algorithms

## 1. Discrete Unknown Layouts

### Mono

```text
x = [ φω ; φγ ]
```

- `φω`: bulk (`ω`) unknowns, one per cut-cell control volume.
- `φγ`: interface (`γ`) unknowns, one per interface control location.

### Two-phase

```text
x = [ φω1 ; φγ1 ; φω2 ; φγ2 ]
```

This ordering is fixed by the public API and used by all assembly and solver paths.

## 2. Core Advection Blocks

`CartesianOperators.jl` provides advection operators from geometry + velocities. Assembly uses:

- bulk advection contributions on `ω` rows,
- interface advection/coupling contributions on `γ` rows.

At high level:

- `ω` rows receive transport operator terms plus source and (for unsteady) mass terms,
- `γ` rows enforce embedded-interface closure selected from local inflow/outflow signs.

## 3. Steady Mono Assembly

The assembled block system has the form:

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

- `A11`, `A12`: bulk advection blocks from cut-cell operators.
- `A21`, `A22`: interface closure rows (`Tγ=g` or `Tγ=Tω` depending on sign).
- Outer BCs are then applied on `ω` rows via advection boundary logic (`Inflow/Outflow/Periodic`).

## 4. Unsteady Mono `θ`-Assembly

For one step `tⁿ -> tⁿ⁺¹`:

1. Assemble steady operator at `tⁿ + θΔt`.
2. If `θ < 1`, add previous-step transport correction on `ω` rows.
3. Add mass diagonal `V/Δt` on `ω` rows.

Equivalent structure:

```math
\left(\frac{V}{\Delta t} + \theta A\right)\phi^{n+1}
=
\frac{V}{\Delta t}\phi^n - (1-\theta)A\phi^n + b.
```

`solve_unsteady!` reuses constant operators/factorizations when matrix and RHS are time-invariant.

## 5. Steady Two-Phase Assembly

The global matrix is assembled in four blocks of unknowns `(ω1, γ1, ω2, γ2)`.

- `ω1`, `ω2` rows: per-phase advection + source + outer BC enforcement.
- `γ1`, `γ2` rows: local interface closure selected from signs `s1 = uγ1·nγ1`, `s2 = uγ2·nγ2`.

Advection needs inflow data only. The implementation does not impose diffusion-style double interface constraints; it enforces one locally well-posed advection closure pattern per interface location.

## 6. Unsteady Two-Phase `θ`-Assembly

The two-phase unsteady step mirrors mono, applied to both phase bulk blocks:

- steady transport evaluated at `t + θΔt`,
- `(1-θ)` previous-step correction on `ω1` and `ω2` rows,
- mass diagonals `V1/Δt` and `V2/Δt` added to `ω1`/`ω2`.

Interface closure rows remain sign-selected at assembly time from interface velocities.

Assumption for constant-operator reuse: velocity/source/boundary callbacks are time-invariant and `dt` is unchanged.

## 7. Regularization and Inactive Rows

Inactive and halo rows are regularized to identity equations.

Why:

- keeps matrix size/layout stable,
- avoids dropping rows/columns across cut-cell topology,
- preserves nonsingularity and robust solves in partially active regions.

## 8. Practical Notes

- `Centered()` is less diffusive and useful for smooth transport where mild dispersive oscillations are acceptable.
- `Upwind1()` is more dissipative but safer around sharp fronts/discontinuities.
- Conservation and sharpness trade off with scheme choice and time step.
- Sign convention is consistent across assembly and solve wrappers; the same inflow-data sign wording is used in tests, theory docs, and README.
