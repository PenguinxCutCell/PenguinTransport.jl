using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

const CX = 0.62
const CY = 0.48
const R = 0.17

const UX = cos(pi / 6)
const UY = sin(pi / 6)

phi(x, y, t) = sin(2pi * (x - UX * t)) * sin(2pi * (y - UY * t))
dphi_dt(x, y, t) = -2pi * UX * cos(2pi * (x - UX * t)) * sin(2pi * (y - UY * t)) -
                   2pi * UY * sin(2pi * (x - UX * t)) * cos(2pi * (y - UY * t))

function build_moments(nx::Int, ny::Int)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    levelset(x, y, _=0.0) = R - sqrt((x - CX)^2 + (y - CY)^2)
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function periodic_adv_bc_2d()
    return CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64), CartesianOperators.AdvPeriodic(Float64)),
        (CartesianOperators.AdvPeriodic(Float64), CartesianOperators.AdvPeriodic(Float64)),
    )
end

function g2l_map(sys)
    g2l = zeros(Int, sys.ops_adv.Nd)
    @inbounds for (l, g) in pairs(sys.dof_omega.indices)
        g2l[g] = l
    end
    return g2l
end

function reduced_from_field(sys, f::Function, t::Float64)
    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = g2l_map(sys)

    u = zeros(Float64, length(sys.dof_omega.indices))
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue

        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        u[l] = f(x, y, t)
    end
    return u
end

function reduced_mass_times_time_derivative(sys, t::Float64)
    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = g2l_map(sys)

    out = zeros(Float64, length(sys.dof_omega.indices))
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue

        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        out[l] = sys.moments.V[g] * dphi_dt(x, y, t)
    end
    return out
end

function circle_band_ids(sys)
    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = g2l_map(sys)

    dx = abs(sys.moments.xyz[1][2] - sys.moments.xyz[1][1])
    dy = abs(sys.moments.xyz[2][2] - sys.moments.xyz[2][1])
    width = 2.0 * max(dx, dy)

    ids = Int[]
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue

        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        dist = abs(sqrt((x - CX)^2 + (y - CY)^2) - R)
        if dist <= width
            push!(ids, l)
        end
    end
    return ids
end

function weighted_norms(sys, e::AbstractVector{Float64}; ids::Union{Nothing,Vector{Int}}=nothing)
    idx = sys.dof_omega.indices
    V = sys.moments.V

    if ids === nothing
        ids = collect(eachindex(e))
    end

    l1 = 0.0
    l2 = 0.0
    linf = 0.0
    w = 0.0
    @inbounds for l in ids
        wi = V[idx[l]]
        ai = abs(e[l])
        l1 += wi * ai
        l2 += wi * ai * ai
        linf = max(linf, ai)
        w += wi
    end

    l1 /= max(w, eps(Float64))
    l2 = sqrt(l2 / max(w, eps(Float64)))
    return l1, l2, linf
end

function run_level(n::Int, tf::Float64)
    moments = build_moments(n, n)

    embedded_exact = function (moms, t)
        dims = (length(moms.xyz[1]), length(moms.xyz[2]))
        li = LinearIndices(dims)
        out = zeros(Float64, prod(dims))
        @inbounds for I in CartesianIndices(dims)
            g = li[I]
            out[g] = phi(moms.xyz[1][I[1]], moms.xyz[2][I[2]], t)
        end
        return out
    end

    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=periodic_adv_bc_2d(),
        scheme=CartesianOperators.MUSCL(CartesianOperators.MC()),
        vel_omega=(UX, UY),
        vel_gamma=(UX, UY),
        embedded_inflow=embedded_exact,
    )
    sys = PenguinTransport.build_system(moments, prob)

    u = reduced_from_field(sys, phi, tf)
    mdt = reduced_mass_times_time_derivative(sys, tf)
    rhs = similar(u)
    PenguinSolverCore.rhs!(rhs, sys, u, nothing, tf)

    defect = rhs .- mdt
    band = circle_band_ids(sys)

    l1, l2, linf = weighted_norms(sys, defect)
    bl1, bl2, blinf = weighted_norms(sys, defect; ids=band)
    return (l1=l1, l2=l2, linf=linf, bl1=bl1, bl2=bl2, blinf=blinf)
end

function main()
    tf = 0.15
    ns = (32, 64, 128)
    errs = [run_level(n, tf) for n in ns]

    println("2D EB manufactured residual consistency (smooth advection)")
    println("  scheme: MUSCL(MC), velocity = (cos 30°, sin 30°), t = $tf")
    println("  geometry: circle obstacle (levelset != -1)")

    for (k, n) in pairs(ns)
        e = errs[k]
        println("  N=$n  L1=$(e.l1)  L2=$(e.l2)  Linf=$(e.linf)  band-L1=$(e.bl1)")
    end

    p_l1 = [log2(errs[i].l1 / errs[i + 1].l1) for i in 1:(length(errs) - 1)]
    p_l2 = [log2(errs[i].l2 / errs[i + 1].l2) for i in 1:(length(errs) - 1)]
    p_bl1 = [log2(errs[i].bl1 / errs[i + 1].bl1) for i in 1:(length(errs) - 1)]

    println("  observed order L1 (global): ", p_l1)
    println("  observed order L2 (global): ", p_l2)
    println("  observed order L1 (EB band): ", p_bl1)
end

main()
