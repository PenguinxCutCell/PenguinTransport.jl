using PenguinTransport
using CartesianGeometry
using CartesianOperators
using CartesianGrids
using PenguinBCs
using Statistics
using Printf

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

function _l2_weighted_error(cap, u_num, u_exact)
    V = cap.buf.V
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    num = 0.0
    den = 0.0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * (u_num[lin] - u_exact[lin])^2
        den += v
    end
    return sqrt(num / max(den, eps(Float64)))
end

_h_from_nodes(x) = minimum(diff(collect(x)))

function _safe_sine(points, c, t)
    out = zeros(Float64, length(points))
    for i in eachindex(points)
        x = points[i][1]
        out[i] = isfinite(x) ? sin(2π * (x - c * t)) : 0.0
    end
    return out
end

function _safe_radial_mode(points; cx=0.5, cy=0.5)
    out = zeros(Float64, length(points))
    for i in eachindex(points)
        x = points[i][1]
        y = length(points[i]) >= 2 ? points[i][2] : NaN
        if isfinite(x) && isfinite(y)
            r = sqrt((x - cx)^2 + (y - cy)^2)
            out[i] = sin(4π * r)
        else
            out[i] = 0.0
        end
    end
    return out
end

function _safe_tangent_velocity(points; cx=0.5, cy=0.5)
    ux = zeros(Float64, length(points))
    uy = zeros(Float64, length(points))
    for i in eachindex(points)
        x = points[i][1]
        y = length(points[i]) >= 2 ? points[i][2] : NaN
        if isfinite(x) && isfinite(y)
            ux[i] = -(y - cy)
            uy[i] =  (x - cx)
        end
    end
    return ux, uy
end

function run_mono_fixed_no_interface(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    grid = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid[1])
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal

    vel = (fill(c, nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    model = TransportModelMono(cap, vel, vel; source=0.0, bc_border=bc, bc_interface=nothing, scheme=scheme)

    u0 = _safe_sine(cap.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady!(model, u0, (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    uf = res.states[end][model.layout.offsets.ω]
    exact = _safe_sine(cap.C_ω, c, tend)
    return _l2_weighted_error(cap, uf, exact)
end

function run_mono_fixed_interface(n, scheme, tscheme)
    c = 1.0
    tend = 0.1
    cx = 0.5
    cy = 0.5
    r0 = 0.22
    grid = (range(0.0, 1.0; length=n), range(0.0, 1.0; length=n))
    h = min(_h_from_nodes(grid[1]), _h_from_nodes(grid[2]))
    body(x, y) = sqrt((x - cx)^2 + (y - cy)^2) - r0
    cap = assembled_capacity(geometric_moments(body, grid, Float64, nan; method=:vofijul); bc=0.0)
    nt = cap.ntotal

    uωx, uωy = _safe_tangent_velocity(cap.C_ω; cx=cx, cy=cy)
    uγx, uγy = _safe_tangent_velocity(cap.C_γ; cx=cx, cy=cy)
    bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
    model = TransportModelMono(cap, (uωx, uωy), (uγx, uγy);
        source=0.0,
        bc_border=bc,
        bc_interface=nothing,
        scheme=scheme,
    )

    u0 = _safe_radial_mode(cap.C_ω; cx=cx, cy=cy)
    umax = max(maximum(abs.(uωx)), maximum(abs.(uωy)))
    dt = 0.4 * h / max(umax, eps(Float64))
    res = solve_unsteady!(model, u0, (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    uf = res.states[end][model.layout.offsets.ω]
    exact = _safe_radial_mode(cap.C_ω; cx=cx, cy=cy)
    return _l2_weighted_error(cap, uf, exact)
end

function run_mono_moving_no_interface(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    grid_nodes = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid_nodes[1])
    cap0 = assembled_capacity(full_moments(grid_nodes); bc=0.0)

    grid = CartesianGrid((0.0,), (1.0,), (n,))
    nt = prod(grid.n)
    vel = (fill(c, nt),)
    model = MovingTransportModelMono(
        grid, (x, t) -> -1.0, vel, vel;
        wγ=(zeros(nt),),
        source=0.0,
        bc_border=BorderConditions(; left=Periodic(), right=Periodic()),
        bc_interface=nothing,
        scheme=scheme,
        geom_method=:vofijul,
    )

    u0 = _safe_sine(cap0.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady_moving!(model, u0, (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    cap = model.cap_slab
    ω = res.states[end][model.layout.offsets.ω]
    exact = _safe_sine(cap.C_ω, c, tend)
    return _l2_weighted_error(cap, ω, exact)
end

function run_mono_moving_material_interface(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    x0 = 0.5
    r = 0.18
    grid_nodes = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid_nodes[1])
    body0(x) = abs(x - x0) - r
    cap0 = assembled_capacity(geometric_moments(body0, grid_nodes, Float64, nan; method=:vofijul); bc=0.0)

    grid = CartesianGrid((0.0,), (1.0,), (n,))
    nt = prod(grid.n)
    vel = (fill(c, nt),)
    body(x, t) = abs(x - (x0 + c * t)) - r
    model = MovingTransportModelMono(
        grid, body, vel, vel;
        wγ=vel,
        source=0.0,
        bc_border=BorderConditions(; left=Periodic(), right=Periodic()),
        bc_interface=nothing,
        scheme=scheme,
        geom_method=:vofijul,
    )

    u0 = _safe_sine(cap0.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady_moving!(model, u0, (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    cap = model.cap_slab
    ω = res.states[end][model.layout.offsets.ω]
    exact = _safe_sine(cap.C_ω, c, tend)
    return _l2_weighted_error(cap, ω, exact)
end

function run_two_fixed_no_interface(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    grid = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid[1])
    cap1 = assembled_capacity(full_moments(grid); bc=0.0)
    cap2 = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap1.ntotal

    vel = (fill(c, nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    model = TransportModelTwoPhase(
        cap1, cap2,
        vel, vel,
        vel, vel;
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=scheme,
    )

    u01 = _safe_sine(cap1.C_ω, c, 0.0)
    u02 = _safe_sine(cap2.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady!(model, (u01, u02), (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    lay = model.layout
    ω1 = res.states[end][lay.ω1]
    ω2 = res.states[end][lay.ω2]
    ex1 = _safe_sine(cap1.C_ω, c, tend)
    ex2 = _safe_sine(cap2.C_ω, c, tend)
    return _l2_weighted_error(cap1, ω1, ex1), _l2_weighted_error(cap2, ω2, ex2)
end

function run_two_fixed_interface_same_scalar(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    x0 = 0.5
    r = 0.18
    grid = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid[1])
    body(x) = abs(x - x0) - r
    cap1 = assembled_capacity(geometric_moments(body, grid, Float64, nan; method=:vofijul); bc=0.0)
    cap2 = assembled_capacity(geometric_moments(x -> -body(x), grid, Float64, nan; method=:vofijul); bc=0.0)
    nt = cap1.ntotal

    vel = (fill(c, nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    model = TransportModelTwoPhase(
        cap1, cap2,
        vel, vel,
        vel, vel;
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=scheme,
    )

    u01 = _safe_sine(cap1.C_ω, c, 0.0)
    u02 = _safe_sine(cap2.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady!(model, (u01, u02), (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    lay = model.layout
    ω1 = res.states[end][lay.ω1]
    ω2 = res.states[end][lay.ω2]
    ex1 = _safe_sine(cap1.C_ω, c, tend)
    ex2 = _safe_sine(cap2.C_ω, c, tend)
    return _l2_weighted_error(cap1, ω1, ex1), _l2_weighted_error(cap2, ω2, ex2)
end

function run_two_moving_no_interface(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    grid_nodes = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid_nodes[1])
    cap0 = assembled_capacity(full_moments(grid_nodes); bc=0.0)

    grid = CartesianGrid((0.0,), (1.0,), (n,))
    nt = prod(grid.n)
    vel = (fill(c, nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    model = MovingTransportModelTwoPhase(
        grid, (x, t) -> -1.0,
        vel, vel,
        vel, vel;
        body2=(x, t) -> -1.0,
        wγ=(zeros(nt),),
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=scheme,
        geom_method=:vofijul,
    )

    u01 = _safe_sine(cap0.C_ω, c, 0.0)
    u02 = _safe_sine(cap0.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady_moving!(model, (u01, u02), (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    cap1 = model.cap1_slab
    cap2 = model.cap2_slab
    lay = model.layout
    ω1 = res.states[end][lay.ω1]
    ω2 = res.states[end][lay.ω2]
    ex1 = _safe_sine(cap1.C_ω, c, tend)
    ex2 = _safe_sine(cap2.C_ω, c, tend)
    return _l2_weighted_error(cap1, ω1, ex1), _l2_weighted_error(cap2, ω2, ex2)
end

function run_two_moving_material_interface_same_scalar(n, scheme, tscheme)
    c = 0.4
    tend = 0.1
    x0 = 0.5
    r = 0.18
    grid_nodes = (range(0.0, 1.0; length=n),)
    h = _h_from_nodes(grid_nodes[1])
    body0(x) = abs(x - x0) - r
    cap10 = assembled_capacity(geometric_moments(body0, grid_nodes, Float64, nan; method=:vofijul); bc=0.0)
    cap20 = assembled_capacity(geometric_moments(x -> -body0(x), grid_nodes, Float64, nan; method=:vofijul); bc=0.0)

    grid = CartesianGrid((0.0,), (1.0,), (n,))
    nt = prod(grid.n)
    vel = (fill(c, nt),)
    body1(x, t) = abs(x - (x0 + c * t)) - r
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    model = MovingTransportModelTwoPhase(
        grid, body1,
        vel, vel,
        vel, vel;
        body2=nothing,
        wγ=vel,
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=scheme,
        geom_method=:vofijul,
    )

    u01 = _safe_sine(cap10.C_ω, c, 0.0)
    u02 = _safe_sine(cap20.C_ω, c, 0.0)
    dt = 0.4 * h / c
    res = solve_unsteady_moving!(model, (u01, u02), (0.0, tend); dt=dt, scheme=tscheme, save_history=false)
    cap1 = model.cap1_slab
    cap2 = model.cap2_slab
    lay = model.layout
    ω1 = res.states[end][lay.ω1]
    ω2 = res.states[end][lay.ω2]
    ex1 = _safe_sine(cap1.C_ω, c, tend)
    ex2 = _safe_sine(cap2.C_ω, c, tend)
    return _l2_weighted_error(cap1, ω1, ex1), _l2_weighted_error(cap2, ω2, ex2)
end

conv(e) = (log(e[1] / e[2]) / log(2), log(e[2] / e[3]) / log(2))
fmt(x) = @sprintf("%.3e", x)
fmto(x) = @sprintf("%.3f", x)

ns = (33, 65, 129)
combos = (
    (Upwind1(), :BE, "Upwind1", "BE"),
    (Upwind1(), :CN, "Upwind1", "CN"),
    (Centered(), :BE, "Centered", "BE"),
    (Centered(), :CN, "Centered", "CN"),
)

println("[MONO]")
for (case, runf) in (
    ("mono_fixed_no_interface", run_mono_fixed_no_interface),
    ("mono_fixed_interface", run_mono_fixed_interface),
    ("mono_moving_no_interface", run_mono_moving_no_interface),
    ("mono_moving_material_interface", run_mono_moving_material_interface),
)
    for (scheme, tscheme, sname, tname) in combos
        e = Float64[]
        for n in ns
            push!(e, runf(n, scheme, tscheme))
        end
        p1, p2 = conv(e)
        println(join([case, sname, tname, fmt(e[1]), fmt(e[2]), fmt(e[3]), fmto(p1), fmto(p2)], "|"))
    end
end

println("[TWO]")
for (case, runf) in (
    ("two_fixed_no_interface", run_two_fixed_no_interface),
    ("two_fixed_interface_same_scalar", run_two_fixed_interface_same_scalar),
    ("two_moving_no_interface", run_two_moving_no_interface),
    ("two_moving_material_interface_same_scalar", run_two_moving_material_interface_same_scalar),
)
    for (scheme, tscheme, sname, tname) in combos
        e1 = Float64[]
        e2 = Float64[]
        for n in ns
            a, b = runf(n, scheme, tscheme)
            push!(e1, a)
            push!(e2, b)
        end
        p11, p12 = conv(e1)
        p21, p22 = conv(e2)
        println(join([
            case, sname, tname,
            fmt(e1[1]), fmt(e1[2]), fmt(e1[3]), fmto(p11), fmto(p12),
            fmt(e2[1]), fmt(e2[2]), fmt(e2[3]), fmto(p21), fmto(p22),
        ], "|"))
    end
end
