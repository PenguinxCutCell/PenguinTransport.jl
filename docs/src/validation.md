# Validation

The test suite includes numerical validation contracts beyond unit checks.

## Conservation

- **Periodic mass conservation**: advection-only runs preserve total mass.
- Checked both semidiscrete residual (`sum(du)`) and time-integrated mass drift.

## Smooth Accuracy

- **Order test on periodic sine wave**:
  - Full-domain setup (`levelset = -1`) with analytical solution.
  - Time integration uses manual mass-matrix-aware SSP RK2 at `CFL = 0.5`.
  - Centered scheme: clear second-order in both `L1` and `L2`.
  - MUSCL (`MC`, `VanLeer`): clear second-order in `L1` (primary FV metric), with lower but convergent `L2` due limiter activation near smooth extrema.

## Nonsmooth Behavior

- **Step-function advection**:
  - Overshoot/undershoot checks for Upwind1 and MUSCL.
  - Guards against unstable limiter/boundary regressions.

## Advection-Diffusion Consistency

- **Manufactured solution test**:
  - Constructs source term so chosen exact field satisfies semidiscrete equation.
  - Verifies final-time relative error against analytic target.

## Performance Gate

- `rhs!` allocation regression test ensures runtime allocations remain small on a representative 1D case.
