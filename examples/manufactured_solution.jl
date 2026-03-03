# ──────────────────────────────────────────────────────────────────
# Manufactured Solution — pure advection with source
#
#   ∂φ/∂t + u⋅∇φ = f
#
# Uses a manufactured analytic solution to verify advection accuracy.
#
# Geometry: [0,1]²,  u = (1, 0.5)
# Exact steady solution: φ(x,y) = sin(2π x) sin(2π y)
# Source f = u⋅∇φ is derived analytically so that the exact solution
# satisfies the steady advection equation.
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra
using SparseArrays

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

omega_view(model, state) = state[model.layout.offsets.ω]

function l1_error(cap, u_num, u_exact)
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

# ─── Manufactured solution parameters ────────────────────────────

# Velocity field
const ux_val = 1.0
const uy_val = 0.5

# φ(x,y) = sin(2πx) sin(2πy)
φ_exact(x, y) = sin(2π * x) * sin(2π * y)

# u⋅∇φ = ux * 2π cos(2πx) sin(2πy) + uy * 2π sin(2πx) cos(2πy)
# Source f = u⋅∇φ balances advection at steady state
source(x, y) = ux_val * 2π * cos(2π * x) * sin(2π * y) +
               uy_val * 2π * sin(2π * x) * cos(2π * y)

# ─── Convergence study: unsteady relaxation to steady state ──────

println("Manufactured solution (advection with source)\n")
println("=== Convergence study: relaxation from φ=0 → steady state ===\n")

dxs = [0.1, 0.05, 0.025]
errors = Float64[]

for dx in dxs
    grid = (0.0:dx:1.0, 0.0:dx:1.0)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal

    bc_adv = BorderConditions(;
        left=Inflow((x, y) -> φ_exact(x, y)),
        right=Outflow(),
        bottom=Inflow((x, y) -> φ_exact(x, y)),
        top=Outflow(),
    )

    uω = (ux_val * ones(nt), uy_val * ones(nt))
    uγ = (ux_val * ones(nt), uy_val * ones(nt))
    model = TransportModelMono(cap, uω, uγ;
        bc_border=bc_adv, source=source, scheme=Upwind1())

    T_relax = 3.0
    dt = 0.5 * dx / (abs(ux_val) + abs(uy_val))
    result = solve_unsteady!(model, zeros(nt), (0.0, T_relax);
        dt=dt, scheme=:BE, save_history=false)
    uf = omega_view(model, result.states[end])

    u_ex = [φ_exact(cap.C_ω[i][1], cap.C_ω[i][2]) for i in 1:nt]
    e = l1_error(cap, uf, u_ex)
    push!(errors, e)
    println("  dx = $dx  →  L1 = $(round(e, sigdigits=4))  (reused=$(result.reused_constant_operator))")
end

println()
for i in 1:length(dxs)-1
    r = dxs[i] / dxs[i+1]
    o = log(errors[i] / errors[i+1]) / log(r)
    println("  $(dxs[i]) → $(dxs[i+1]):  order = $(round(o, digits=2))")
end

# ─── Centered scheme comparison (periodic BCs) ──────────────────

println("\n=== Centered scheme with periodic BCs (dx=0.05, CN) ===\n")

dx = 0.05
grid = (0.0:dx:1.0, 0.0:dx:1.0)
cap = assembled_capacity(full_moments(grid); bc=0.0)
nt = cap.ntotal

# For periodic BCs with centered scheme, use a simple translation test
bc_per = BorderConditions(;
    left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

# Source f = u⋅∇φ so that φ(x,y) = sin(2πx) sin(2πy) is the steady solution
uω_c = (ux_val * ones(nt), uy_val * ones(nt))
uγ_c = (ux_val * ones(nt), uy_val * ones(nt))
model_c = TransportModelMono(cap, uω_c, uγ_c;
    bc_border=bc_per, source=source, scheme=Centered())

T_relax = 3.0
dt = 0.5 * dx / (abs(ux_val) + abs(uy_val))
u0_c = [φ_exact(cap.C_ω[i][1], cap.C_ω[i][2]) for i in 1:nt]
result_c = solve_unsteady!(model_c, u0_c, (0.0, T_relax);
    dt=dt, scheme=:CN, save_history=false)
uf_c = omega_view(model_c, result_c.states[end])
u_ex = [φ_exact(cap.C_ω[i][1], cap.C_ω[i][2]) for i in 1:nt]
e_c = l1_error(cap, uf_c, u_ex)
println("  L1 error (centered+CN, periodic, init=exact): $(round(e_c, sigdigits=4))")
println("  Reused constant operator: $(result_c.reused_constant_operator)")
