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

function smooth_initial(moments, dof_omega; mean=0.6, amp=0.4)
    xcells = moments.xyz[1]
    u0 = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        x = xcells[idx]
        u0[i] = mean + amp * sin(2pi * x)
    end
    return u0
end

function exact_sine_advection_diffusion(moments, dof_omega; t, vel, kappa, mean=0.6, amp=0.4)
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

moments = periodic_1d_moments()
bc_adv = CartesianOperators.AdvBoxBC(
    (CartesianOperators.AdvPeriodic(Float64),),
    (CartesianOperators.AdvPeriodic(Float64),),
)
vel = 0.75
kappa = 0.01
tf = 0.5
prob = PenguinTransport.TransportProblem(;
    kappa=kappa,
    bc_adv=bc_adv,
    scheme=CartesianOperators.Centered(),
    vel_omega=vel,
    vel_gamma=vel,
)
sys = PenguinTransport.build_system(moments, prob)
u0 = smooth_initial(moments, sys.dof_omega)
dt = PenguinTransport.cfl_dt(sys, u0; cfl=0.45)

odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, tf); p=nothing)
sol = SciMLBase.solve(odeprob, OrdinaryDiffEq.Rosenbrock23(autodiff=false);
    adaptive=false,
    dt=dt,
    saveat=tf,
)

u_end = sol.u[end]
u_ex = exact_sine_advection_diffusion(moments, sys.dof_omega; t=tf, vel=vel, kappa=kappa)
l2rel, linf = weighted_errors(sys, u_end, u_ex)

println("1D advection-diffusion analytic check at t=$tf")
println("  CFL dt: ", dt)
println("  L2(rel) error: ", l2rel)
println("  Linf error   : ", linf)
println("  Mean(num/ex) : ", sum(u_end) / length(u_end), " / ", sum(u_ex) / length(u_ex))
