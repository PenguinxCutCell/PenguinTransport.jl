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

function gaussian_on_active(moments, dof_omega; x0=0.2, sigma=0.06)
    xcells = moments.xyz[1]
    u0 = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        xc = xcells[idx]
        u0[i] = exp(-((xc - x0)^2) / (2 * sigma^2))
    end
    return u0
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
    moments = periodic_1d_moments()
    sys_adv = build_advection_system(moments; vel=1.0)
    sys_dif = build_diffusion_system(moments; kappa=0.01)

    u = gaussian_on_active(moments, sys_adv.dof_omega)
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

    println("Transport + diffusion Strang splitting")
    println("  dt = ", dt, ", nsteps = ", nsteps)
    println("  Final reduced-state norm: ", norm(u))
end

main()
