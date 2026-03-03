# ──────────────────────────────────────────────────────────────────
# Sharp Peak Advection — upwind vs centered comparison
#
#   ∂φ/∂t + u⋅∇φ = 0,   u = (1,),   periodic BCs,   1D
#
# Compares numerical dissipation (upwind) vs oscillation (centered)
# when advecting a sharp (narrow) Gaussian peak.
# ──────────────────────────────────────────────────────────────────

using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore
using LinearAlgebra

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

omega_view(model, state) = state[model.layout.offsets.ω]

function metrics(cap, u_num, u_exact)
    V = cap.buf.V
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    l1 = 0.0; linf = 0.0; den = 0.0
    umin = Inf; umax = -Inf
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        err_i = abs(u_num[lin] - u_exact[lin])
        l1 += v * err_i
        linf = max(linf, err_i)
        den += v
        umin = min(umin, u_num[lin])
        umax = max(umax, u_num[lin])
    end
    return (l1=l1/den, linf=linf, umin=umin, umax=umax)
end

# ─── Run ──────────────────────────────────────────────────────────

dx = 0.02
grid = (0.0:dx:1.0,)
cap = assembled_capacity(full_moments(grid); bc=0.0)
nt = cap.ntotal
x = cap.xyz[1]

# Very sharp Gaussian (width σ ≈ 2.5 cells)
sigma = 0.05
u0 = exp.(-((x .- 0.5).^2) ./ (2 * sigma^2))
u_exact = copy(u0)   # exact after T = 1

bc = BorderConditions(; left=Periodic(), right=Periodic())
T_end = 1.0
dt = 0.5 * dx

println("Sharp peak advection — upwind vs centered (1D, dx=$dx, T=$T_end)\n")

for (label, scheme, tscheme) in [
    ("Upwind1 + BE",   Upwind1(),  :BE),
    ("Centered + CN",  Centered(), :CN),
]
    model = TransportModelMono(cap, (ones(nt),), (ones(nt),); bc_border=bc, scheme=scheme)
    result = solve_unsteady!(model, u0, (0.0, T_end); dt=dt, scheme=tscheme, save_history=false)
    uf = omega_view(model, result.states[end])
    m = metrics(cap, uf, u_exact)

    println("  $label:")
    println("    L1 error     = $(round(m.l1, sigdigits=4))")
    println("    L∞ error     = $(round(m.linf, sigdigits=4))")
    println("    u_min        = $(round(m.umin, sigdigits=4))  (exact: 0)")
    println("    u_max        = $(round(m.umax, sigdigits=4))  (exact: 1)")
    println("    monotonicity = $(m.umin >= -1e-10 && m.umax <= 1.0 + 1e-10 ? "✓ preserved" : "✗ violated")")
    println()
end
