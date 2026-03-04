# API

## Exported Symbols

```@docs
TransportModelMono
assemble_steady_mono!
assemble_unsteady_mono!
solve_steady!
solve_unsteady!
update_advection_ops!
rebuild!
```

## Notes

- Unsteady solver time schemes: `:BE`, `:CN`, or numeric `theta`.
- Spatial advection schemes are selected through the `scheme` field of `TransportModelMono` (for example `Centered()` or `Upwind1()`).
- Embedded interface closure is sign-based on `uγ·nγ`: inflow (`< 0`) can impose `Tγ=g` via `bc_interface`, otherwise continuity (`Tγ=Tω`) is used.
- No-flow mode is recovered by setting interface velocity input to zero, e.g. `uγ = (zeros(nt), zeros(nt))` in 2D.
