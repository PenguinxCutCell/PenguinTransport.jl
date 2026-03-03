# ──────────────────────────────────────────────────────────────────
# Smooth Blob Translation — constant-velocity advection of a Gaussian
#
#   ∂φ/∂t + u⋅∇φ = 0,   u = (1, 0),   periodic BCs
#
# Verifies L1/L2 error decay after one full period (T = 1).
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

function _l1_error(cap, u_num, u_exact)
    V = cap.buf.V
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    num = 0.0; den = 0.0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * abs(u_num[lin] - u_exact[lin])
        den += v
    end
    return num / den
end

function _l2_error(cap, u_num, u_exact)
    V = cap.buf.V
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    num = 0.0; den = 0.0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * (u_num[lin] - u_exact[lin])^2
        den += v
    end
    return sqrt(num / den)
end

omega_view(model, state) = state[model.layout.offsets.ω]

# ─── Run ──────────────────────────────────────────────────────────

println("Smooth blob translation in 2D — convergence study\n")

bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
T_end = 1.0
sigma = 0.1
cx0, cy0 = 0.3, 0.5

dxs = [0.1, 0.05, 0.025]
l1_errors = Float64[]
l2_errors = Float64[]

for dx in dxs
    grid = (0.0:dx:1.0, 0.0:dx:1.0)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal
    x = [cap.C_ω[i][1] for i in 1:nt]
    y = [cap.C_ω[i][2] for i in 1:nt]

    # Gaussian blob
    u0 = exp.(-((x .- cx0).^2 .+ (y .- cy0).^2) ./ (2 * sigma^2))
    u_exact = copy(u0)   # exact after period T=1 with u=(1,0)

    uω = (ones(nt), zeros(nt))
    uγ = (ones(nt), zeros(nt))
    dt = 0.5 * dx

    model = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Centered())
    result = solve_unsteady!(model, u0, (0.0, T_end); dt=dt, scheme=:CN, save_history=false)
    uf = omega_view(model, result.states[end])

    e1 = _l1_error(cap, uf, u_exact)
    e2 = _l2_error(cap, uf, u_exact)
    push!(l1_errors, e1)
    push!(l2_errors, e2)
    println("  dx = $dx  →  L1 = $(round(e1, sigdigits=4))  L2 = $(round(e2, sigdigits=4))")
end

println()
for i in 1:length(dxs)-1
    r = dxs[i] / dxs[i+1]
    o1 = log(l1_errors[i] / l1_errors[i+1]) / log(r)
    o2 = log(l2_errors[i] / l2_errors[i+1]) / log(r)
    println("  $(dxs[i]) → $(dxs[i+1]):  L1 order = $(round(o1, digits=2)),  L2 order = $(round(o2, digits=2))")
end
