using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using LinearAlgebra

const CX = 0.62
const CY = 0.48
const R = 0.17

function build_moments(nx::Int=48, ny::Int=48)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    levelset(x, y, _=0.0) = R - sqrt((x - CX)^2 + (y - CY)^2)
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function rk2_step!(u, sys, dt, t)
    idx = sys.dof_omega.indices
    V = sys.moments.V

    rhs0 = similar(u)
    rhs1 = similar(u)
    u1 = similar(u)

    PenguinSolverCore.rhs!(rhs0, sys, u, nothing, t)
    @inbounds for i in eachindex(u)
        u1[i] = u[i] + dt * rhs0[i] / V[idx[i]]
    end

    PenguinSolverCore.rhs!(rhs1, sys, u1, nothing, t + dt)
    @inbounds for i in eachindex(u)
        u[i] = 0.5 * u[i] + 0.5 * (u1[i] + dt * rhs1[i] / V[idx[i]])
    end

    return u
end

function main()
    phi0 = 0.37
    vel = (1.0, 0.7)

    moments = build_moments()
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvInflow(phi0), CartesianOperators.AdvInflow(phi0)),
        (CartesianOperators.AdvOutflow(Float64), CartesianOperators.AdvOutflow(Float64)),
    )

    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=bc_adv,
        scheme=CartesianOperators.Upwind1(),
        vel_omega=vel,
        vel_gamma=vel,
        embedded_inflow=phi0,
    )

    sys = PenguinTransport.build_system(moments, prob)
    u = fill(phi0, length(sys.dof_omega.indices))

    tf = 0.8
    dt_cfl = PenguinTransport.cfl_dt(sys, u; cfl=0.45, include_diffusion=false)
    nsteps = max(1, ceil(Int, tf / dt_cfl))
    dt = tf / nsteps

    t = 0.0
    for _ in 1:nsteps
        rk2_step!(u, sys, dt, t)
        t += dt
    end

    err_all = maximum(abs.(u .- phi0))

    dims = sys.ops_adv.dims
    li = LinearIndices(dims)
    g2l = zeros(Int, sys.ops_adv.Nd)
    @inbounds for (l, g) in pairs(sys.dof_omega.indices)
        g2l[g] = l
    end

    dx = abs(sys.moments.xyz[1][2] - sys.moments.xyz[1][1])
    dy = abs(sys.moments.xyz[2][2] - sys.moments.xyz[2][1])
    width = 2.0 * max(dx, dy)

    err_band = 0.0
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        l = g2l[g]
        l == 0 && continue

        x = sys.moments.xyz[1][I[1]]
        y = sys.moments.xyz[2][I[2]]
        dist = abs(sqrt((x - CX)^2 + (y - CY)^2) - R)
        if dist <= width
            err_band = max(err_band, abs(u[l] - phi0))
        end
    end

    println("2D EB freestream preservation (diagonal inflow)")
    println("  grid          : $(dims[1]-1)x$(dims[2]-1)")
    println("  steps, dt     : $nsteps, $dt")
    println("  max |phi-phi0| (all fluid): $err_all")
    println("  max |phi-phi0| (EB band)  : $err_band")
    println("  finite state  : ", all(isfinite, u))
end

main()
