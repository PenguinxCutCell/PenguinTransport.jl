@testset "Performance: rhs! allocations" begin
    sys = build_periodic_1d_system(;
        nx=128,
        scheme=CartesianOperators.Upwind1(),
        vel=1.0,
        kappa=0.02,
    )
    u = randn(length(sys.dof_omega.indices))
    du = zeros(Float64, length(u))

    # Warm-up call to avoid compilation allocations in the measurement.
    PenguinSolverCore.rhs!(du, sys, u, nothing, 0.0)
    alloc = @allocated PenguinSolverCore.rhs!(du, sys, u, nothing, 0.1)

    @test alloc <= 2048
end
