@testset "RHS sanity" begin
    @testset "Zero velocity gives zero rhs (pure advection)" begin
        sys = build_cut_system(; kappa=0.0, vel_omega=(0.0, 0.0), vel_gamma=(0.0, 0.0), source=nothing)
        u = randn(length(sys.dof_omega.indices))
        du = zeros(Float64, length(u))
        PenguinSolverCore.rhs!(du, sys, u, nothing, 0.0)
        @test isapprox(norm(du), 0.0; atol=0.0, rtol=0.0)
    end

    @testset "Periodic constant-state rhs is finite" begin
        sys = build_periodic_1d_system(; scheme=CartesianOperators.MUSCL(CartesianOperators.Minmod()), vel=1.0, kappa=0.0)
        u = fill(2.0, length(sys.dof_omega.indices))
        du = zeros(Float64, length(u))
        PenguinSolverCore.rhs!(du, sys, u, nothing, 0.1)

        @test length(du) == length(u)
        @test all(isfinite, du)

        Tω, Tγ = PenguinTransport.full_state(sys, u)
        @test length(Tω) == sys.ops_diff.Nd
        @test length(Tγ) == sys.ops_diff.Nd
        @test isapprox(Tω, Tγ; atol=0.0, rtol=0.0)
    end
end
