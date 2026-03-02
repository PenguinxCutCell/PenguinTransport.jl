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

@inline periodic_dist(x::Float64, c::Float64) = abs(mod(x - c + 0.5, 1.0) - 0.5)

function exact_gaussian_advected(moments, dof_omega; vel::Float64, t::Float64, x0::Float64, sigma::Float64)
    xcells = moments.xyz[1]
    uex = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        x = xcells[idx] + vel * t
        d = periodic_dist(x, x0)
        uex[i] = exp(-(d^2) / (2 * sigma^2))
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
    l2rel = sqrt(num / max(den, eps(Float64)))
    return l2rel, linf
end

function solve_scheme(scheme; vel=1.0, tf=0.3, x0=0.25, sigma=0.06)
    moments = periodic_1d_moments()
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
    sys = PenguinTransport.build_system(moments, prob)
    u0 = gaussian_on_active(moments, sys.dof_omega; x0=x0, sigma=sigma)
    dt = PenguinTransport.cfl_dt(sys, u0; cfl=0.5)
    odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, tf); p=nothing)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        adaptive=false,
        dt=dt,
        saveat=tf,
    )
    u_num = sol.u[end]
    u_ex = exact_gaussian_advected(moments, sys.dof_omega; vel=vel, t=tf, x0=x0, sigma=sigma)
    l2rel, linf = weighted_errors(sys, u_num, u_ex)
    return sys, u_num, u_ex, l2rel, linf
end

tf = 0.3
sys_up, u_up, uex, l2_up, linf_up = solve_scheme(CartesianOperators.Upwind1(); tf=tf)
sys_mu, u_mu, _, l2_mu, linf_mu = solve_scheme(CartesianOperators.MUSCL(CartesianOperators.Minmod()); tf=tf)

println("Periodic 1D advection at t=$tf")
println("  Upwind1: L2(rel) = $l2_up, Linf = $linf_up")
println("  MUSCL  : L2(rel) = $l2_mu, Linf = $linf_mu")
println("  ||MUSCL - Upwind1||2: ", norm(u_mu - u_up))

Tω_up, _ = PenguinTransport.full_state(sys_up, u_up)
Tω_mu, _ = PenguinTransport.full_state(sys_mu, u_mu)
println("  Sample exact/reduced (idx 10): exact=$(uex[10]), Upwind1=$(u_up[10]), MUSCL=$(u_mu[10])")
println("  Full-state sample (cell 10): Upwind1=$(Tω_up[10]), MUSCL=$(Tω_mu[10])")
