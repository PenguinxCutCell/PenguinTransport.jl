using Test
using LinearAlgebra

@testset "Embedded inflow BC forwarding" begin
    vel = (1.0, 0.0)

    sys_none = build_cut_system(; kappa=0.0, scheme=CartesianOperators.Upwind1(), vel_omega=vel, vel_gamma=vel, embedded_inflow=nothing)
    sys_emb = build_cut_system(; kappa=0.0, scheme=CartesianOperators.Upwind1(), vel_omega=vel, vel_gamma=vel, embedded_inflow=2.0)

    nω = length(sys_emb.dof_omega.indices)
    u0 = zeros(Float64, nω)
    du_none = similar(u0)
    du_emb = similar(u0)

    PenguinSolverCore.rhs!(du_none, sys_none, u0, nothing, 0.0)
    PenguinSolverCore.rhs!(du_emb, sys_emb, u0, nothing, 0.0)

    @test norm(du_none) ≤ 1e-12
    @test norm(du_emb) > 1e-10

    du_ref_full = zeros(Float64, sys_emb.ops_adv.Nd)
    CartesianOperators.convection!(
        du_ref_full,
        sys_emb.ops_adv,
        sys_emb.uω_full,
        sys_emb.uγ_full,
        sys_emb.Tω_full,
        sys_emb.Tγ_full,
        sys_emb.work_adv;
        scheme=sys_emb.scheme,
        moments=sys_emb.moments,
        embedded_bc=sys_emb.embedded_bc,
        t=0.0,
    )

    idx = sys_emb.dof_omega.indices
    @test all(i -> isapprox(du_emb[i], du_ref_full[idx[i]]; atol=1e-12, rtol=1e-12), eachindex(idx))
end

@testset "Embedded inflow callback payload" begin
    vel = (0.7, -0.2)
    inflow_fun = (moms, t) -> 1.0 + 0.1 * t

    sys = build_cut_system(; kappa=0.0, scheme=CartesianOperators.Centered(), vel_omega=vel, vel_gamma=vel, embedded_inflow=inflow_fun)
    u0 = zeros(Float64, length(sys.dof_omega.indices))
    du = similar(u0)

    PenguinSolverCore.rhs!(du, sys, u0, nothing, 0.25)
    @test all(isfinite, du)
end
