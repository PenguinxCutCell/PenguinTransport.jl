using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using LinearAlgebra

function full_2d_moments(nx::Int=24, ny::Int=24)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    full_domain(_x, _y, _=0.0) = -1.0
    return CartesianGeometry.geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)
end

function rotating_velocity(moments)
    dims = (length(moments.xyz[1]), length(moments.xyz[2]))
    li = LinearIndices(dims)
    Nd = prod(dims)
    u1 = zeros(Float64, Nd)
    u2 = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        x = moments.xyz[1][I[1]] - 0.5
        y = moments.xyz[2][I[2]] - 0.5
        u1[idx] = -2pi * y
        u2[idx] = 2pi * x
    end
    return (u1, u2)
end

function gaussian_blob(moments, dof_omega; x0=0.62, y0=0.5, sigma=0.05)
    dims = (length(moments.xyz[1]), length(moments.xyz[2]))
    li = LinearIndices(dims)
    field = zeros(Float64, prod(dims))
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        x = moments.xyz[1][I[1]]
        y = moments.xyz[2][I[2]]
        field[idx] = exp(-((x - x0)^2 + (y - y0)^2) / (2 * sigma^2))
    end
    return field[dof_omega.indices]
end

function exact_rotated_gaussian(moments, dof_omega; t, x0=0.62, y0=0.5, sigma=0.05, cx=0.5, cy=0.5, omega=2pi)
    theta = -omega * t
    ct = cos(theta)
    st = sin(theta)
    xr = x0 - cx
    yr = y0 - cy
    xc_t = cx + ct * xr - st * yr
    yc_t = cy + st * xr + ct * yr

    dims = (length(moments.xyz[1]), length(moments.xyz[2]))
    li = LinearIndices(dims)
    full = zeros(Float64, prod(dims))
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        x = moments.xyz[1][I[1]]
        y = moments.xyz[2][I[2]]
        full[idx] = exp(-((x - xc_t)^2 + (y - yc_t)^2) / (2 * sigma^2))
    end
    return full[dof_omega.indices]
end

function weighted_errors(sys, u_num, u_ex)
    idx = sys.dof_omega.indices
    V = sys.moments.V
    num = 0.0
    den = 0.0
    linf = 0.0
    @inbounds for i in eachindex(idx)
        w = V[idx[i]]
        e = u_num[i] - u_ex[i]
        num += w * e * e
        den += w * u_ex[i] * u_ex[i]
        linf = max(linf, abs(e))
    end
    return sqrt(num / max(den, eps(Float64))), linf
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
    moments = full_2d_moments()
    vel = rotating_velocity(moments)
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        scheme=CartesianOperators.Upwind1(),
        vel_omega=vel,
        vel_gamma=vel,
    )
    sys = PenguinTransport.build_system(moments, prob)
    u0 = gaussian_blob(moments, sys.dof_omega)
    tf = 0.02
    dt = PenguinTransport.cfl_dt(sys, u0; cfl=0.2, include_diffusion=false)
    u = copy(u0)
    t = 0.0
    nsteps = max(1, ceil(Int, tf / dt))
    dt = tf / nsteps
    for _ in 1:nsteps
        rk2_step!(u, sys, dt, t)
        t += dt
    end

    u_ex = exact_rotated_gaussian(moments, sys.dof_omega; t=t)
    l2rel, linf = weighted_errors(sys, u, u_ex)

    println("2D rotating transport")
    println("  dt used      : ", dt)
    println("  Initial norm: ", norm(u0))
    println("  Final norm:   ", norm(u))
    println("  Change norm:  ", norm(u - u0))
    println("  Exact check L2(rel): ", l2rel)
    println("  Exact check Linf   : ", linf)
end

main()
