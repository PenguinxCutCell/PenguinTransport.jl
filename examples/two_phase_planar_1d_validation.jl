# ──────────────────────────────────────────────────────────────────
# Two-Phase Planar 1D Validation
#
# Validation layers in 1D with interface at x = a:
# 1) zero-source case (u1 != u2): check interface flux ratio and conservation
# 2) source in phase 1 only: check interface values/flux shift consistently
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra

function planar_moments_left(grid; x0=0.47)
    levelset(x, y=0, z=0) = x - x0
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function planar_moments_right(grid; x0=0.47)
    levelset(x, y=0, z=0) = x0 - x
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function interface_flux_metrics(model::TransportModelTwoPhase, x)
    cap1 = model.cap1
    cap2 = model.cap2
    lay = model.layout
    nt = cap1.ntotal
    κ1 = PenguinTransport._interface_flux_diag(model.ops1)
    κ2 = PenguinTransport._interface_flux_diag(model.ops2)
    num_abs = 0.0
    den_abs = 0.0
    niface = 0
    for i in 1:nt
        Γ = 0.5 * (cap1.buf.Γ[i] + cap2.buf.Γ[i])
        (isfinite(Γ) && Γ > 0) || continue
        T1γ = x[lay.γ1[i]]
        T2γ = x[lay.γ2[i]]
        r = κ1[i] * T1γ + κ2[i] * T2γ
        num_abs += abs(r)
        den_abs += abs(κ1[i] * T1γ) + abs(κ2[i] * T2γ)
        niface += 1
    end
    return (rel_abs=num_abs / (den_abs + eps(Float64)), niface=niface)
end

function active_phase1_upstream_error(cap1, w1, x0, g)
    LI = LinearIndices(cap1.nnodes)
    err = 0.0
    used = 0
    for I in CartesianIndices(cap1.nnodes)
        lin = LI[I]
        I[1] == cap1.nnodes[1] && continue
        v = cap1.buf.V[lin]
        (isfinite(v) && v > 0) || continue
        x = cap1.C_ω[lin][1]
        x <= x0 - 0.1 || continue
        err = max(err, abs(w1[lin] - g))
        used += 1
    end
    return err, used
end

println("Two-phase planar 1D validation\n")

x0 = 0.47
grid = (0.0:0.025:1.0,)
cap1 = assembled_capacity(planar_moments_left(grid; x0=x0); bc=0.0)
cap2 = assembled_capacity(planar_moments_right(grid; x0=x0); bc=0.0)
nt = cap1.ntotal
u1 = 1.0
u2 = 2.0
g = 1.3

# Case A: zero sources (unsteady BE end state for robust solvability)
model0 = TransportModelTwoPhase(
    cap1, cap2,
    (u1 .* ones(nt),), (u1 .* ones(nt),),
    (u2 .* ones(nt),), (u2 .* ones(nt),);
    source1=0.0,
    source2=0.0,
    bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
    bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
    scheme=Upwind1(),
)
res0 = solve_unsteady!(model0, (zeros(nt), zeros(nt)), (0.0, 2.0); dt=0.01, scheme=:BE, save_history=false)
x0_state = res0.states[end]
lay = model0.layout
iface = findall(i -> cap1.buf.Γ[i] > 0 || cap2.buf.Γ[i] > 0, 1:nt)
i = iface[1]
κ1 = PenguinTransport._interface_flux_diag(model0.ops1)
κ2 = PenguinTransport._interface_flux_diag(model0.ops2)
T1γ0 = x0_state[lay.γ1[i]]
T2γ0 = x0_state[lay.γ2[i]]
met0 = interface_flux_metrics(model0, x0_state)
err_up, n_up = active_phase1_upstream_error(cap1, x0_state[lay.ω1], x0, g)

println("Case A: zero source")
println("  interface cells                    : ", met0.niface)
println("  interface flux rel. residual       : ", met0.rel_abs)
println("  ratio T2γ/T1γ                      : ", T2γ0 / T1γ0)
println("  expected ratio -(κ1/κ2)            : ", -(κ1[i] / κ2[i]))
println("  upstream phase-1 |ω1 - g|_L∞       : ", err_up, " (", n_up, " cells)")
println()

# Case B: source in phase 1 only
σ = 0.4
modelσ = TransportModelTwoPhase(
    cap1, cap2,
    (u1 .* ones(nt),), (u1 .* ones(nt),),
    (u2 .* ones(nt),), (u2 .* ones(nt),);
    source1=σ,
    source2=0.0,
    bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
    bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
    scheme=Upwind1(),
)
resσ = solve_unsteady!(modelσ, (zeros(nt), zeros(nt)), (0.0, 2.0); dt=0.01, scheme=:BE, save_history=false)
xσ_state = resσ.states[end]
T1γσ = xσ_state[lay.γ1[i]]
T2γσ = xσ_state[lay.γ2[i]]
metσ = interface_flux_metrics(modelσ, xσ_state)

println("Case B: source1 = $σ, source2 = 0")
println("  interface flux rel. residual       : ", metσ.rel_abs)
println("  ΔT1γ (with source - no source)     : ", T1γσ - T1γ0)
println("  ΔT2γ (with source - no source)     : ", T2γσ - T2γ0)

@assert met0.rel_abs < 1e-12
@assert metσ.rel_abs < 1e-12
@assert abs(T2γ0 - (-(κ1[i] / κ2[i])) * T1γ0) < 1e-12
@assert T1γσ > T1γ0
@assert T2γσ > T2γ0

println("\nValidation passed.")
