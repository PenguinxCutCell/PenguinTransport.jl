using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using SciMLBase
using OrdinaryDiffEq
using LinearAlgebra

function periodic_1d_moments(nx::Int=120)
    x = collect(range(0.0, 1.0; length=nx + 1))
    full_domain(_x, _=0.0) = -1.0
    return CartesianGeometry.geometric_moments(full_domain, (x,), Float64, zero; method=:implicitintegration)
end

function build_advection_system(moments; vel=1.0, scheme=CartesianOperators.MUSCL(CartesianOperators.MC()))
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64),),
        (CartesianOperators.AdvPeriodic(Float64),),
    )
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=bc_adv,
        scheme=scheme,
        vel_omega=vel,
        vel_gamma=vel,
    )
    return PenguinTransport.build_system(moments, prob)
end

function build_diffusion_system(moments; kappa=0.01)
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64),),
        (CartesianOperators.AdvPeriodic(Float64),),
    )
    prob = PenguinTransport.TransportProblem(;
        kappa=kappa,
        bc_adv=bc_adv,
        scheme=CartesianOperators.Centered(),
        vel_omega=0.0,
        vel_gamma=0.0,
    )
    return PenguinTransport.build_system(moments, prob)
end

function sine_on_active(moments, dof_omega; mean=0.6, amp=0.35)
    xcells = moments.xyz[1]
    u0 = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        x = xcells[idx]
        u0[i] = mean + amp * sin(2pi * x)
    end
    return u0
end

function exact_sine_advection_diffusion(moments, dof_omega; t, vel, kappa, mean=0.6, amp=0.35)
    xcells = moments.xyz[1]
    decay = exp(-4pi^2 * kappa * t)
    uex = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        x = xcells[idx]
        uex[i] = mean + amp * decay * sin(2pi * (x + vel * t))
    end
    return uex
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

function step_subproblem(sys, u, t0, dt)
    prob = PenguinSolverCore.sciml_odeproblem(sys, u, (t0, t0 + dt); p=nothing)
    sol = SciMLBase.solve(
        prob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        adaptive=false,
        dt=dt,
        saveat=[t0 + dt],
    )
    return sol.u[end]
end

function main()
    vel = 1.0
    kappa = 0.01
    moments = periodic_1d_moments()
    sys_adv = build_advection_system(moments; vel=vel)
    sys_dif = build_diffusion_system(moments; kappa=kappa)

    u = sine_on_active(moments, sys_adv.dof_omega)
    dt = min(
        PenguinTransport.cfl_dt(sys_adv, u; cfl=0.45, include_diffusion=false),
        PenguinTransport.cfl_dt(sys_dif, u; cfl=0.45, include_diffusion=true),
    )
    t = 0.0
    tf = 0.25
    nsteps = Int(round(tf / dt))
    dt = tf / nsteps

    for _ in 1:nsteps
        u = step_subproblem(sys_dif, u, t, 0.5 * dt)
        u = step_subproblem(sys_adv, u, t + 0.5 * dt, dt)
        u = step_subproblem(sys_dif, u, t + 1.5 * dt, 0.5 * dt)
        t += dt
    end

    u_ex = exact_sine_advection_diffusion(moments, sys_adv.dof_omega; t=tf, vel=vel, kappa=kappa)
    l2rel, linf = weighted_errors(sys_adv, u, u_ex)

    println("Transport + diffusion Strang splitting (analytic check)")
    println("  dt = ", dt, ", nsteps = ", nsteps)
    println("  L2(rel) error: ", l2rel)
    println("  Linf error   : ", linf)
end

main()
