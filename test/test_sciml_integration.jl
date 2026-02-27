if Base.find_package("SciMLBase") === nothing || Base.find_package("OrdinaryDiffEq") === nothing
    @testset "SciML integration (guarded)" begin
        @test true
    end
else
    using SciMLBase
    using OrdinaryDiffEq

    supports_sciml = let ok = true
        try
            sys_probe = build_cut_system(; kappa=0.0)
            u0_probe = zeros(Float64, length(sys_probe.dof_omega.indices))
            PenguinSolverCore.sciml_odeproblem(sys_probe, u0_probe, (0.0, 0.1); p=nothing)
        catch err
            if err isa ArgumentError && occursin("requires SciMLBase", sprint(showerror, err))
                ok = false
            else
                rethrow(err)
            end
        end
        ok
    end

    if !supports_sciml
        @testset "SciML integration (guarded)" begin
            @test true
        end
    else
        @testset "SciML integration (guarded)" begin
            tstar = 0.37

            sys = build_cut_system(; kappa=0.0)
            u0 = zeros(Float64, length(sys.dof_omega.indices))
            supd = PenguinTransport.SourceUpdater((_sys, _u, _p, _t) -> 1.0)
            PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([tstar]), supd)

            prob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, 1.0); p=nothing)
            alg = OrdinaryDiffEq.Rosenbrock23(autodiff=false)
            sol = SciMLBase.solve(prob, alg;
                reltol=1e-9, abstol=1e-9, save_everystep=true,
            )

            @test any(t -> isapprox(t, tstar; atol=1000eps(tstar)), sol.t)
            @test sys.rebuild_calls == 0

            sys_ref = build_cut_system(; kappa=0.0)
            prob_ref = PenguinSolverCore.sciml_odeproblem(sys_ref, u0, (0.0, 1.0); p=nothing)
            sol_ref = SciMLBase.solve(prob_ref, alg; reltol=1e-9, abstol=1e-9)

            @test norm(sol.u[end] - sol_ref.u[end]) > 1e-9
        end
    end
end
