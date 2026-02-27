# API Reference

## Core Types

```@docs
TransportProblem
TransportSystem
```

## Build and State

```@docs
build_system
full_state
cfl_dt
```

## Time-Varying Updaters

```@docs
KappaUpdater
SchemeUpdater
VelocityUpdater
AdvBCUpdater
SourceUpdater
```

## Steady Matrix-Free Solve

```@docs
steady_linear_problem
steady_solve
```

## `PenguinSolverCore` Integration Points

The package implements:

- `PenguinSolverCore.rhs!(du, sys, u, p, t)`
- `PenguinSolverCore.mass_matrix(sys)`
- `PenguinSolverCore.rebuild!(sys, u, p, t)`

and uses `UpdateManager` schedules through `PenguinSolverCore.apply_scheduled_updates!`.
