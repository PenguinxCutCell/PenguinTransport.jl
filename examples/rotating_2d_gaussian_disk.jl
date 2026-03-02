using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

function disk_moments(; nx::Int=48, ny::Int=48, cx::Float64=0.5, cy::Float64=0.5, radius::Float64=0.33)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    levelset(x, y, _=0.0) = sqrt((x - cx)^2 + (y - cy)^2) - radius
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function solid_body_velocity(moments; cx::Float64=0.5, cy::Float64=0.5, omega::Float64=2pi)
    dims = (length(moments.xyz[1]), length(moments.xyz[2]))
    li = LinearIndices(dims)
    Nd = prod(dims)
    u1 = zeros(Float64, Nd)
    u2 = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        x = moments.xyz[1][I[1]] - cx
        y = moments.xyz[2][I[2]] - cy
        u1[g] = -omega * y
        u2[g] = omega * x
    end
    return (u1, u2)
end

function gaussian_reduced(moments, dof_omega; t::Float64=0.0, x0::Float64=0.67, y0::Float64=0.5, sigma::Float64=0.06, cx::Float64=0.5, cy::Float64=0.5, omega::Float64=2pi)
    theta = -omega * t
    ct = cos(theta)
    st = sin(theta)
    xr = x0 - cx
    yr = y0 - cy
    xc = cx + ct * xr - st * yr
    yc = cy + st * xr + ct * yr

    dims = (length(moments.xyz[1]), length(moments.xyz[2]))
    li = LinearIndices(dims)
    full = zeros(Float64, prod(dims))
    @inbounds for I in CartesianIndices(dims)
        g = li[I]
        x = moments.xyz[1][I[1]]
        y = moments.xyz[2][I[2]]
        full[g] = exp(-((x - xc)^2 + (y - yc)^2) / (2 * sigma^2))
    end
    return full[dof_omega.indices]
end

function rk2_step!(u, sys, dt::Float64, t::Float64)
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

function weighted_errors(sys, u_num, u_ref)
    idx = sys.dof_omega.indices
    V = sys.moments.V
    num = 0.0
    den = 0.0
    linf = 0.0
    @inbounds for i in eachindex(u_num)
        e = u_num[i] - u_ref[i]
        w = V[idx[i]]
        num += w * e * e
        den += w * u_ref[i] * u_ref[i]
        linf = max(linf, abs(e))
    end
    return sqrt(num / max(den, eps(Float64))), linf
end

function main()
    moments = disk_moments()
    vel = solid_body_velocity(moments)

    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        scheme=CartesianOperators.Centered(),
        vel_omega=vel,
        vel_gamma=vel,
        embedded_inflow=nothing, # Embedded boundary: do nothing.
    )
    sys = PenguinTransport.build_system(moments, prob)

    u0 = gaussian_reduced(moments, sys.dof_omega; t=0.0)
    u = copy(u0)

    tf = 0.01
    dt_cfl = PenguinTransport.cfl_dt(sys, u; cfl=0.3, include_diffusion=false)
    nsteps = max(1, ceil(Int, tf / dt_cfl))
    dt = tf / nsteps

    t = 0.0
    for _ in 1:nsteps
        rk2_step!(u, sys, dt, t)
        t += dt
    end

    u_exact = gaussian_reduced(moments, sys.dof_omega; t=t)
    l2rel, linf = weighted_errors(sys, u, u_exact)

    println("2D rotating transport in a disk (embedded boundary do-nothing)")
    println("  active omega cells : ", length(sys.dof_omega.indices))
    println("  dt / steps         : ", dt, " / ", nsteps)
    println("  t_final            : ", t)
    println("  relative L2 error  : ", l2rel)
    println("  Linf error         : ", linf)
end

main()
