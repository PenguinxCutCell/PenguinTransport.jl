if Base.find_package("SciMLBase") === nothing || Base.find_package("OrdinaryDiffEq") === nothing
    @testset "Validation: manufactured advection-diffusion (guarded)" begin
        @test true
    end
else
    using SciMLBase
    using OrdinaryDiffEq

    function _exact_profile(sys, t)
        idx = sys.dof_omega.indices
        x = sys.moments.xyz[1]
        out = zeros(Float64, length(idx))
        et = exp(-t)
        @inbounds for i in eachindex(idx)
            ξ = x[idx[i]]
            out[i] = 1.0 + 0.2 * sin(2pi * ξ) * et
        end
        return out
    end

    function _exact_dudt(sys, t)
        idx = sys.dof_omega.indices
        x = sys.moments.xyz[1]
        out = zeros(Float64, length(idx))
        et = exp(-t)
        @inbounds for i in eachindex(idx)
            ξ = x[idx[i]]
            out[i] = -0.2 * sin(2pi * ξ) * et
        end
        return out
    end

    @testset "Validation: manufactured advection-diffusion" begin
        nx = 96
        vel = 0.7
        κ = 0.03
        scheme = CartesianOperators.Centered()

        sys_op = build_periodic_1d_system(; nx=nx, scheme=scheme, vel=vel, kappa=κ, source=nothing)
        sys = build_periodic_1d_system(; nx=nx, scheme=scheme, vel=vel, kappa=κ, source=nothing)

        idx = sys.dof_omega.indices
        V = sys.moments.V[idx]
        tmp_rhs = zeros(Float64, length(idx))

        sys.sourcefun = function (_sys, _u, _p, t)
            u_ex = _exact_profile(sys, t)
            dudt_ex = _exact_dudt(sys, t)

            PenguinSolverCore.rhs!(tmp_rhs, sys_op, u_ex, nothing, t)
            src = similar(tmp_rhs)
            @inbounds for i in eachindex(src)
                src[i] = (V[i] * dudt_ex[i] - tmp_rhs[i]) / V[i]
            end
            return src
        end

        u0 = _exact_profile(sys, 0.0)
        tf = 0.25
        prob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, tf); p=nothing)
        sol = SciMLBase.solve(
            prob,
            OrdinaryDiffEq.Rosenbrock23(autodiff=false);
            saveat=[tf],
            reltol=1e-10,
            abstol=1e-10,
        )
        @test SciMLBase.successful_retcode(sol)

        u_ref = _exact_profile(sys, tf)
        rel = norm(sol.u[end] - u_ref) / max(norm(u_ref), eps(Float64))
        @test rel < 1e-7
    end
end
