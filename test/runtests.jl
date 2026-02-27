using Test
using LinearAlgebra
using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

function build_cut_moments()
    x = collect(range(0.0, 1.0; length=8))
    y = collect(range(0.0, 1.0; length=7))
    levelset(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.28
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function build_full_1d_moments(; nx=40)
    full_domain(_x, _=0.0) = -1.0
    x = collect(range(0.0, 1.0; length=nx + 1))
    return CartesianGeometry.geometric_moments(full_domain, (x,), Float64, zero; method=:implicitintegration)
end

function build_cut_system(;
    kappa=0.0,
    scheme=CartesianOperators.Centered(),
    vel_omega=(0.0, 0.0),
    vel_gamma=vel_omega,
    source=nothing,
    bc_adv=nothing,
    bc_diff=nothing,
)
    moments = build_cut_moments()
    prob = PenguinTransport.TransportProblem(;
        kappa=kappa,
        bc_diff=bc_diff,
        bc_adv=bc_adv,
        scheme=scheme,
        vel_omega=vel_omega,
        vel_gamma=vel_gamma,
        source=source,
    )
    return PenguinTransport.build_system(moments, prob)
end

function build_periodic_1d_system(;
    nx::Int=80,
    scheme=CartesianOperators.Upwind1(),
    vel=1.0,
    kappa=0.0,
    source=nothing,
)
    moments = build_full_1d_moments(; nx=nx)
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64),),
        (CartesianOperators.AdvPeriodic(Float64),),
    )
    prob = PenguinTransport.TransportProblem(;
        kappa=kappa,
        bc_adv=bc_adv,
        scheme=scheme,
        vel_omega=vel,
        vel_gamma=vel,
        source=source,
    )
    return PenguinTransport.build_system(moments, prob)
end

include("test_masking.jl")
include("test_rhs.jl")
include("test_updates_and_rebuild.jl")
include("test_sciml_integration.jl")
include("test_steady_solver.jl")
include("test_validation_mass.jl")
include("test_validation_order.jl")
include("test_validation_boundedness.jl")
include("test_validation_manufactured_advection_diffusion.jl")
include("test_performance_rhs_allocations.jl")
