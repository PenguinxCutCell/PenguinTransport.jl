using CartesianGeometry: nan
using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinTransport

function active_mask(cap)
    N = length(cap.nnodes)
    LI = LinearIndices(cap.nnodes)
    m = falses(cap.ntotal)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = cap.buf.V[lin]
        m[lin] = isfinite(v) && v > 0
    end
    return m
end

function weighted_l2(cap, u, uex)
    N = length(cap.nnodes)
    LI = LinearIndices(cap.nnodes)
    num = 0.0
    den = 0.0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = cap.buf.V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * (u[lin] - uex[lin])^2
        den += v
    end
    return sqrt(num / max(den, eps(Float64)))
end

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (65, 65))
nt = prod(grid.n)
U = 0.3
Tend = 0.2

body(x, y, t) = sqrt((x - (0.35 + U * t))^2 + (y - 0.5)^2) - 0.2

uω = (fill(U, nt), zeros(nt))
uγ = (fill(U, nt), zeros(nt))
wγ = (fill(U, nt), zeros(nt))
bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())

model = MovingTransportModelMono(
    grid, body, uω, uγ;
    wγ=wγ,
    source=0.0,
    bc_border=bc,
    bc_interface=nothing,
    scheme=Upwind1(),
)

x = collect(range(0.0, 1.0; length=grid.n[1]))
y = collect(range(0.0, 1.0; length=grid.n[2]))
u0 = zeros(Float64, nt)
LI = LinearIndices((grid.n...,))
for I in CartesianIndices((grid.n...,))
    lin = LI[I]
    u0[lin] = sin(2pi * x[I[1]]) * cos(2pi * y[I[2]])
end

dx = 1.0 / (grid.n[1] - 1)
dt = 0.4 * dx / U
res = solve_unsteady_moving!(model, u0, (0.0, Tend); dt=dt, scheme=:BE, save_history=false)

cap = model.cap_slab
@assert !(cap === nothing)
ω = res.states[end][model.layout.offsets.ω]
uex = [sin(2pi * (cap.C_ω[i][1] - U * Tend)) * cos(2pi * cap.C_ω[i][2]) for i in 1:nt]
err = weighted_l2(cap, ω, uex)

println("moving mono material translation")
println("  grid         = ", grid.n)
println("  final time   = ", Tend)
println("  weighted L2  = ", err)
println("  active cells = ", count(active_mask(cap)))
