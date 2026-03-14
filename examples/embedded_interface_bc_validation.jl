# ──────────────────────────────────────────────────────────────────
# Embedded Interface BC Validation (sign-based inflow/outflow)
#
# Validates interface closure on a cut-cell geometry using the same
# discrete embedded-interface coefficient κ carried by transport assembly:
#   - if κ < 0 (interface inflow), enforce Tγ = g
#   - otherwise (outflow / near-zero), enforce Tγ = Tω
#
# We report weighted L2/L∞ residual norms for both conditions and
# compare two cases:
#   1) bc_interface = Inflow(g)    -> inflow residual should be tiny
#   2) bc_interface = nothing      -> inflow residual vs g should be large
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra

function obstacle_moments(grid; r=0.22, cx=0.5, cy=0.5)
    # omega = outside disk (embedded obstacle)
    levelset(x, y, _=0) = r - sqrt((x - cx)^2 + (y - cy)^2)
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

"""
Compute interface residual norms from a solved state.

Returns weighted L2 and L∞ residuals for:
- inflow condition residual `Tγ - g` on cells where discrete `κ < 0`,
- outflow closure residual `Tγ - Tω` on cells where `κ ≥ 0`.
"""
function interface_residuals(model, state, gfun)
    cap = model.cap
    lay = model.layout.offsets
    Tω = state[lay.ω]
    Tγ = state[lay.γ]
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    κ = PenguinTransport._interface_flux_diag(model.ops)
    tolκ = 100 * eps(Float64)

    num_in = 0.0
    den_in = 0.0
    linf_in = 0.0
    nin = 0

    num_out = 0.0
    den_out = 0.0
    linf_out = 0.0
    nout = 0

    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        Γ = cap.buf.Γ[lin]
        (isfinite(Γ) && Γ > 0) || continue

        κi = κ[lin]
        inflow = κi < -tolκ * max(1.0, abs(κi))
        if inflow
            g = gfun(cap.C_γ[lin][1], cap.C_γ[lin][2])
            r = Tγ[lin] - g
            num_in += Γ * r^2
            den_in += Γ
            linf_in = max(linf_in, abs(r))
            nin += 1
        else
            r = Tγ[lin] - Tω[lin]
            num_out += Γ * r^2
            den_out += Γ
            linf_out = max(linf_out, abs(r))
            nout += 1
        end
    end

    return (
        l2_in=den_in > 0 ? sqrt(num_in / den_in) : 0.0,
        linf_in=linf_in,
        nin=nin,
        l2_out=den_out > 0 ? sqrt(num_out / den_out) : 0.0,
        linf_out=linf_out,
        nout=nout,
    )
end

function run_case(cap, uω, uγ, bc_border, bc_interface, gfun)
    model = TransportModelMono(
        cap,
        uω,
        uγ;
        bc_border=bc_border,
        bc_interface=bc_interface,
        scheme=Upwind1(),
    )
    nt = cap.ntotal
    res = solve_unsteady!(model, zeros(nt), (0.0, 1.0); dt=0.02, scheme=:BE, save_history=false)
    return interface_residuals(model, res.states[end], gfun)
end

println("Embedded interface BC validation (cut-cell geometry)\n")

dx = 0.05
grid = (0.0:dx:1.0, 0.0:dx:1.0)
cap = assembled_capacity(obstacle_moments(grid); bc=0.0)
nt = cap.ntotal

uω = (ones(nt), zeros(nt))
uγ = (ones(nt), zeros(nt))

gfun(x, y) = 1.0 + 0.2 * sin(2π * y)
bc_border = BorderConditions(;
    left=Inflow((x, y) -> gfun(x, y)),
    right=Outflow(),
    bottom=Outflow(),
    top=Outflow(),
)

with_inflow = run_case(cap, uω, uγ, bc_border, Inflow((x, y) -> gfun(x, y)), gfun)
without_inflow = run_case(cap, uω, uγ, bc_border, nothing, gfun)

println("Case A: bc_interface = Inflow(g)")
println("  inflow cells:  $(with_inflow.nin)")
println("  outflow cells: $(with_inflow.nout)")
println("  ||Tγ-g||_L2(Γ-)      = $(with_inflow.l2_in)")
println("  ||Tγ-g||_L∞(Γ-)      = $(with_inflow.linf_in)")
println("  ||Tγ-Tω||_L2(Γ+)     = $(with_inflow.l2_out)")
println("  ||Tγ-Tω||_L∞(Γ+)     = $(with_inflow.linf_out)")
println()

println("Case B: bc_interface = nothing")
println("  ||Tγ-g||_L2(Γ-)      = $(without_inflow.l2_in)")
println("  ||Tγ-g||_L∞(Γ-)      = $(without_inflow.linf_in)")
println("  ||Tγ-Tω||_L2(Γ+)     = $(without_inflow.l2_out)")
println("  ||Tγ-Tω||_L∞(Γ+)     = $(without_inflow.linf_out)")

# Validation checks
@assert with_inflow.nin > 0 && with_inflow.nout > 0
@assert with_inflow.l2_in < 1e-8
@assert with_inflow.l2_out < 1e-8
@assert without_inflow.l2_in > 1e-3
@assert without_inflow.l2_out < 1e-8

println("\nValidation passed.")
