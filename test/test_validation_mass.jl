if Base.find_package("SciMLBase") === nothing || Base.find_package("OrdinaryDiffEq") === nothing
    @testset "Validation: mass conservation (guarded)" begin
        @test true
    end
else
    using SciMLBase
    using OrdinaryDiffEq

    function _mass(sys, u)
        idx = sys.dof_omega.indices
        V = sys.moments.V
        s = 0.0
        @inbounds for i in eachindex(idx)
            s += V[idx[i]] * u[i]
        end
        return s
    end

    @testset "Validation: mass conservation (periodic advection)" begin
        sys = build_periodic_1d_system(;
            nx=96,
            scheme=CartesianOperators.Upwind1(),
            vel=1.0,
            kappa=0.0,
        )
        n = length(sys.dof_omega.indices)
        u0 = randn(n)

        du = zeros(Float64, n)
        PenguinSolverCore.rhs!(du, sys, u0, nothing, 0.0)
        @test abs(sum(du)) < 1e-11

        dt = PenguinTransport.cfl_dt(sys, u0; cfl=0.5)
        @test isfinite(dt)
        @test dt > 0.0

        prob = PenguinSolverCore.sciml_odeproblem(sys, copy(u0), (0.0, 0.3); p=nothing)
        sol = SciMLBase.solve(
            prob,
            OrdinaryDiffEq.Rosenbrock23(autodiff=false);
            adaptive=false,
            dt=dt,
            saveat=[0.3],
            reltol=1e-12,
            abstol=1e-12,
        )
        @test SciMLBase.successful_retcode(sol)

        m0 = _mass(sys, u0)
        m1 = _mass(sys, sol.u[end])
        @test abs(m1 - m0) < 1e-10 * max(1.0, abs(m0))
    end
end
