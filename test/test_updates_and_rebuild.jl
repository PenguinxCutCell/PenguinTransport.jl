@testset "Update + rebuild contract" begin
    @testset "KappaUpdater is rhs_only (no rebuild)" begin
        sys = build_cut_system(; kappa=0.0)
        u = zeros(Float64, length(sys.dof_omega.indices))

        kupd = PenguinTransport.KappaUpdater((_sys, _u, _p, _t) -> 1.75)
        PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.5]), kupd)
        PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.5; step=0)

        @test sys.rebuild_calls == 0
        @test isapprox(sys.kappa, 1.75; atol=0.0, rtol=0.0)
    end

    @testset "AdvBCUpdater triggers rebuild on periodicity change" begin
        sys = build_cut_system(; kappa=0.0)
        u = zeros(Float64, length(sys.dof_omega.indices))
        @test sys.adv_periodic == (false, false)

        bc_periodic_x = CartesianOperators.AdvBoxBC(
            (CartesianOperators.AdvPeriodic(Float64), CartesianOperators.AdvOutflow(Float64)),
            (CartesianOperators.AdvPeriodic(Float64), CartesianOperators.AdvOutflow(Float64)),
        )
        upd = PenguinTransport.AdvBCUpdater((_sys, _u, _p, _t) -> bc_periodic_x)
        PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.25]), upd)
        PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.25; step=0)

        @test sys.rebuild_calls == 1
        @test sys.ops_dirty == false
        @test sys.adv_periodic == (true, false)
    end
end
