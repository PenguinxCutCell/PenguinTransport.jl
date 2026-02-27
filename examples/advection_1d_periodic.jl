using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using SciMLBase
using OrdinaryDiffEq
using LinearAlgebra

function periodic_1d_moments(nx::Int=200)
    x = collect(range(0.0, 1.0; length=nx + 1))
    full_domain(_x, _=0.0) = -1.0
    return CartesianGeometry.geometric_moments(full_domain, (x,), Float64, zero; method=:implicitintegration)
end

function gaussian_on_active(moments, dof_omega; x0=0.25, sigma=0.06)
    xcells = moments.xyz[1]
    u0 = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        xc = xcells[idx]
        u0[i] = exp(-((xc - x0)^2) / (2 * sigma^2))
    end
    return u0
end

function solve_scheme(scheme)
    moments = periodic_1d_moments()
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64),),
        (CartesianOperators.AdvPeriodic(Float64),),
    )
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=bc_adv,
        scheme=scheme,
        vel_omega=1.0,
        vel_gamma=1.0,
    )
    sys = PenguinTransport.build_system(moments, prob)
    u0 = gaussian_on_active(moments, sys.dof_omega)
    dt = PenguinTransport.cfl_dt(sys, u0; cfl=0.5)
    odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, 0.3); p=nothing)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        adaptive=false,
        dt=dt,
        saveat=0.3,
    )
    return sys, sol.u[end]
end

sys_up, u_up = solve_scheme(CartesianOperators.Upwind1())
sys_mu, u_mu = solve_scheme(CartesianOperators.MUSCL(CartesianOperators.Minmod()))

println("Periodic 1D advection at t=0.3")
println("  Upwind1 mass-weighted norm: ", norm(u_up))
println("  MUSCL   mass-weighted norm: ", norm(u_mu))
println("  Difference norm (MUSCL - Upwind1): ", norm(u_mu - u_up))

Tω_up, _ = PenguinTransport.full_state(sys_up, u_up)
Tω_mu, _ = PenguinTransport.full_state(sys_mu, u_mu)
println("  Full-state sample (cell 10): Upwind1=$(Tω_up[10]), MUSCL=$(Tω_mu[10])")
