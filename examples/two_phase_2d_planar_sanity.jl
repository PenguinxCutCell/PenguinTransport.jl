# ──────────────────────────────────────────────────────────────────
# Two-Phase 2D Planar Interface Sanity
#
# Runs an unsteady two-phase transport case on a 2D planar interface
# and reports:
# - finite-value sanity,
# - interface flux conservation residual.
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra

function planar_moments_left(grid; x0=0.47)
    levelset(x, y, _=0) = x - x0
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function planar_moments_right(grid; x0=0.47)
    levelset(x, y, _=0) = x0 - x
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function interface_flux_rel(model::TransportModelTwoPhase, x)
    cap1 = model.cap1
    cap2 = model.cap2
    lay = model.layout
    nt = cap1.ntotal
    num = 0.0
    den = 0.0
    niface = 0
    for i in 1:nt
        Γ = 0.5 * (cap1.buf.Γ[i] + cap2.buf.Γ[i])
        (isfinite(Γ) && Γ > 0) || continue
        s1 = model.u1γ[1][i] * cap1.n_γ[i][1] + model.u1γ[2][i] * cap1.n_γ[i][2]
        s2 = model.u2γ[1][i] * cap2.n_γ[i][1] + model.u2γ[2][i] * cap2.n_γ[i][2]
        T1γ = x[lay.γ1[i]]
        T2γ = x[lay.γ2[i]]
        r = Γ * (s1 * T1γ + s2 * T2γ)
        num += abs(r)
        den += abs(Γ * s1 * T1γ) + abs(Γ * s2 * T2γ)
        niface += 1
    end
    return (rel=num / (den + eps(Float64)), niface=niface)
end

println("Two-phase 2D planar interface sanity\n")

x0 = 0.47
grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
cap1 = assembled_capacity(planar_moments_left(grid; x0=x0); bc=0.0)
cap2 = assembled_capacity(planar_moments_right(grid; x0=x0); bc=0.0)
nt = cap1.ntotal

model = TransportModelTwoPhase(
    cap1, cap2,
    (ones(nt), zeros(nt)), (ones(nt), zeros(nt)),
    (2.0 .* ones(nt), zeros(nt)), (2.0 .* ones(nt), zeros(nt));
    source1=0.0,
    source2=0.0,
    bc_border1=BorderConditions(; left=Inflow(1.0), right=Outflow(), bottom=Periodic(), top=Periodic()),
    bc_border2=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Periodic(), top=Periodic()),
    scheme=Upwind1(),
)

res = solve_unsteady!(model, (zeros(nt), zeros(nt)), (0.0, 0.2); dt=0.02, scheme=:BE, save_history=false)
xf = res.states[end]
met = interface_flux_rel(model, xf)

println("  system size                         : ", length(xf))
println("  all finite                          : ", all(isfinite, xf))
println("  interface cells                     : ", met.niface)
println("  interface flux relative residual    : ", met.rel)
println("  reused constant operator            : ", res.reused_constant_operator)

@assert all(isfinite, xf)
@assert met.niface > 0
@assert met.rel < 1e-12

println("\nSanity checks passed.")
