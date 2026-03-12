using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinTransport

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

function run_case(n)
    grid = CartesianGrid((0.0,), (1.0,), (n,))
    nt = prod(grid.n)
    c = 0.4
    a0 = 0.47
    Tend = 0.1

    body1(x, t) = x - (a0 + c * t)
    vel = (fill(c, nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())

    model = MovingTransportModelTwoPhase(
        grid, body1, vel, vel, vel, vel;
        wγ=vel,
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=Upwind1(),
    )

    x = collect(range(0.0, 1.0; length=n))
    u01 = sin.(2pi .* x)
    u02 = cos.(2pi .* x)
    dt = 0.4 * (1.0 / (n - 1)) / c
    res = solve_unsteady_moving!(model, (u01, u02), (0.0, Tend); dt=dt, scheme=:BE, save_history=false)

    cap1 = model.cap1_slab
    cap2 = model.cap2_slab
    @assert !(cap1 === nothing)
    @assert !(cap2 === nothing)

    lay = model.layout
    ω1 = res.states[end][lay.ω1]
    ω2 = res.states[end][lay.ω2]
    ex1 = [sin(2pi * (cap1.C_ω[i][1] - c * Tend)) for i in 1:nt]
    ex2 = [cos(2pi * (cap2.C_ω[i][1] - c * Tend)) for i in 1:nt]
    return weighted_l2(cap1, ω1, ex1), weighted_l2(cap2, ω2, ex2)
end

ns = (33, 65, 129)
errs = [run_case(n) for n in ns]

e1 = [e[1] for e in errs]
e2 = [e[2] for e in errs]
ord1 = log(e1[1] / e1[2]) / log(2)
ord2 = log(e2[1] / e2[2]) / log(2)

println("moving two-phase planar translation")
for (n, e) in zip(ns, errs)
    println("  n=$n: err1=$(e[1]) err2=$(e[2])")
end
println("  estimated order (phase 1, first refinement) = ", ord1)
println("  estimated order (phase 2, first refinement) = ", ord2)
println("  interface classification uses λk = (ukγ - wγ)·nkγ")
