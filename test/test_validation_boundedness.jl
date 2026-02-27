if Base.find_package("SciMLBase") === nothing || Base.find_package("OrdinaryDiffEq") === nothing
    @testset "Validation: boundedness (guarded)" begin
        @test true
    end
else
    using SciMLBase
    using OrdinaryDiffEq

    function _step_state(sys)
        idx = sys.dof_omega.indices
        x = sys.moments.xyz[1]
        u = zeros(Float64, length(idx))
        @inbounds for i in eachindex(idx)
            u[i] = x[idx[i]] < 0.5 ? 1.0 : 0.0
        end
        return u
    end

    function _solve_step(sys, u0; tf=0.2)
        prob = PenguinSolverCore.sciml_odeproblem(sys, copy(u0), (0.0, tf); p=nothing)
        return SciMLBase.solve(
            prob,
            OrdinaryDiffEq.Rosenbrock23(autodiff=false);
            saveat=[tf],
            reltol=1e-10,
            abstol=1e-10,
        )
    end

    @testset "Validation: boundedness / overshoot control" begin
        sys_up = build_periodic_1d_system(; nx=120, scheme=CartesianOperators.Upwind1(), vel=1.0, kappa=0.0)
        u0 = _step_state(sys_up)
        sol_up = _solve_step(sys_up, u0)
        @test SciMLBase.successful_retcode(sol_up)
        u_up = sol_up.u[end]

        up_overshoot = maximum(u_up) - 1.0
        up_undershoot = -minimum(u_up)
        @test up_overshoot < 1e-2
        @test up_undershoot < 1e-2

        sys_muscl = build_periodic_1d_system(;
            nx=120,
            scheme=CartesianOperators.MUSCL(CartesianOperators.Minmod()),
            vel=1.0,
            kappa=0.0,
        )
        sol_m = _solve_step(sys_muscl, u0)
        @test SciMLBase.successful_retcode(sol_m)
        u_m = sol_m.u[end]

        muscl_overshoot = maximum(u_m) - 1.0
        muscl_undershoot = -minimum(u_m)
        @test muscl_overshoot < 5e-2
        @test muscl_undershoot < 5e-2
    end
end
