@setup_workload begin
    x = collect(range(0.0, 1.0; length=18))
    full_domain(_x, _=0.0) = -1.0
    moments = CartesianGeometry.geometric_moments(full_domain, (x,), Float64, zero; method=:implicitintegration)
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64),),
        (CartesianOperators.AdvPeriodic(Float64),),
    )
    prob = TransportProblem(;
        kappa=0.01,
        bc_adv=bc_adv,
        scheme=CartesianOperators.Upwind1(),
        vel_omega=1.0,
        vel_gamma=1.0,
        source=0.0,
    )
    sys = build_system(moments, prob)
    u0 = zeros(Float64, length(sys.dof_omega.indices))
    du = similar(u0)

    @compile_workload begin
        PenguinSolverCore.rhs!(du, sys, u0, nothing, 0.0)
        cfl_dt(sys, u0; cfl=0.5)
        steady_linear_problem(sys; u0=u0)
    end
end
