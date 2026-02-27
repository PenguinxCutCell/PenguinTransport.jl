using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using SciMLBase
using OrdinaryDiffEq
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

function gaussian_blob(moments, dof_omega; x0=0.7, y0=0.5, sigma=0.07)
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

odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, 0.1); p=nothing)
sol = SciMLBase.solve(
    odeprob,
    OrdinaryDiffEq.Rosenbrock23(autodiff=false);
    reltol=1e-6,
    abstol=1e-6,
    saveat=[0.0, 0.1],
)

println("2D rotating transport")
println("  Initial norm: ", norm(sol.u[1]))
println("  Final norm:   ", norm(sol.u[end]))
println("  Change norm:  ", norm(sol.u[end] - sol.u[1]))
