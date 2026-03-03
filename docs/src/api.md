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
- Embedded interface runs use no-flow mode (`u·n=0`) by setting interface velocity input to zero, e.g. `uγ = (zeros(nt), zeros(nt))` in 2D.
