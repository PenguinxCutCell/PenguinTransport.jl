using LinearSolve
using SciMLBase

function build_steady_test_system(; source=0.0, vel=0.35, kappa=0.08)
    moments = build_full_1d_moments(; nx=36)
    bc_diff = CartesianOperators.BoxBC(
        (CartesianOperators.Dirichlet(1.0),),
        (CartesianOperators.Dirichlet(0.0),),
    )
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvOutflow(Float64),),
        (CartesianOperators.AdvOutflow(Float64),),
    )
    prob = PenguinTransport.TransportProblem(;
        kappa=kappa,
        bc_diff=bc_diff,
        bc_adv=bc_adv,
        scheme=CartesianOperators.Upwind1(),
        vel_omega=vel,
        vel_gamma=vel,
        source=source,
    )
    return PenguinTransport.build_system(moments, prob)
end

@testset "steady_solve matrix-free residual contract" begin
    sys = build_steady_test_system(; source=0.25, vel=0.3, kappa=0.07)

    sol = PenguinTransport.steady_solve(
        sys;
        alg=LinearSolve.SimpleGMRES(),
        abstol=1e-10,
        reltol=1e-10,
        maxiters=20_000,
    )

    @test SciMLBase.successful_retcode(sol)

    res = zeros(Float64, length(sol.u))
    PenguinSolverCore.rhs!(res, sys, sol.u, nothing, 0.0)
    @test norm(res) < 1e-8
end

@testset "steady_linear_problem accepts time-dependent source callback" begin
    sys = build_steady_test_system(; source=(sys, u, p, t) -> p.shift + t, vel=0.2, kappa=0.09)
    p = (shift=0.15,)

    sol_t0 = PenguinTransport.steady_solve(
        sys;
        p=p,
        t=0.0,
        alg=LinearSolve.SimpleGMRES(),
        abstol=1e-10,
        reltol=1e-10,
        maxiters=20_000,
    )
    sol_t1 = PenguinTransport.steady_solve(
        sys;
        p=p,
        t=0.6,
        alg=LinearSolve.SimpleGMRES(),
        abstol=1e-10,
        reltol=1e-10,
        maxiters=20_000,
    )

    @test SciMLBase.successful_retcode(sol_t0)
    @test SciMLBase.successful_retcode(sol_t1)
    @test norm(sol_t0.u - sol_t1.u) > 1e-8
end
