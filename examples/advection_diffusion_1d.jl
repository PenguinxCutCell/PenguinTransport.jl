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

function smooth_initial(moments, dof_omega)
    xcells = moments.xyz[1]
    u0 = zeros(Float64, length(dof_omega.indices))
    @inbounds for i in eachindex(dof_omega.indices)
        idx = dof_omega.indices[i]
        x = xcells[idx]
        u0[i] = 0.6 + 0.4 * sin(2pi * x)
    end
    return u0
end

moments = periodic_1d_moments()
bc_adv = CartesianOperators.AdvBoxBC(
    (CartesianOperators.AdvPeriodic(Float64),),
    (CartesianOperators.AdvPeriodic(Float64),),
)
prob = PenguinTransport.TransportProblem(;
    kappa=0.01,
    bc_adv=bc_adv,
    scheme=CartesianOperators.Centered(),
    vel_omega=0.75,
    vel_gamma=0.75,
)
sys = PenguinTransport.build_system(moments, prob)
u0 = smooth_initial(moments, sys.dof_omega)
dt = PenguinTransport.cfl_dt(sys, u0; cfl=0.45)

odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, 0.5); p=nothing)
sol = SciMLBase.solve(odeprob, OrdinaryDiffEq.Rosenbrock23(autodiff=false);
    adaptive=false,
    dt=dt,
    saveat=0.5,
)

u_end = sol.u[end]
println("1D advection-diffusion with kappa=0.01 at t=0.5")
println("  CFL dt: ", dt)
println("  Reduced-state norm: ", norm(u_end))
println("  Mean value: ", sum(u_end) / length(u_end))
