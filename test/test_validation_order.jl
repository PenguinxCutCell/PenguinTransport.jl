function _sine_state(sys, t, vel)
    idx = sys.dof_omega.indices
    x = sys.moments.xyz[1]
    u = zeros(Float64, length(idx))
    @inbounds for i in eachindex(idx)
        ξ = x[idx[i]] + vel * t
        u[i] = sin(2pi * ξ)
    end
    return u
end

function _l2_weighted_error(sys, u_num, u_ref)
    idx = sys.dof_omega.indices
    V = sys.moments.V
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(idx)
        w = V[idx[i]]
        e = u_num[i] - u_ref[i]
        num += w * e * e
        den += w
    end
    return sqrt(num / den)
end

function _l1_weighted_error(sys, u_num, u_ref)
    idx = sys.dof_omega.indices
    V = sys.moments.V
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(idx)
        w = V[idx[i]]
        num += w * abs(u_num[i] - u_ref[i])
        den += w
    end
    return num / den
end

function _rk2_ssp_integrate(
    sys::TransportSystem{N,T},
    u0::AbstractVector{T},
    t_final::Real;
    cfl::Real=0.5,
    p=nothing,
    t0::Real=0.0,
) where {N,T}
    t_final > 0 || throw(ArgumentError("t_final must be positive"))
    cfl > 0 || throw(ArgumentError("cfl must be positive"))

    idx = sys.dof_omega.indices
    V = sys.moments.V
    n = length(idx)
    length(u0) == n || throw(DimensionMismatch("u0 has length $(length(u0)); expected $n"))

    dt_cfl = PenguinTransport.cfl_dt(sys, u0; cfl=cfl, p=p, t=t0, include_diffusion=false)
    isfinite(dt_cfl) || throw(ArgumentError("non-finite CFL timestep for RK2 integration"))
    dt_cfl > zero(T) || throw(ArgumentError("non-positive CFL timestep for RK2 integration"))

    nsteps = max(1, ceil(Int, t_final / dt_cfl))
    dt = convert(T, t_final / nsteps)

    u = copy(u0)
    u1 = similar(u)
    rhs0 = similar(u)
    rhs1 = similar(u)

    t = convert(T, t0)
    @inbounds for _ in 1:nsteps
        PenguinSolverCore.rhs!(rhs0, sys, u, p, t)
        for i in eachindex(u)
            u1[i] = u[i] + dt * rhs0[i] / V[idx[i]]
        end

        PenguinSolverCore.rhs!(rhs1, sys, u1, p, t + dt)
        for i in eachindex(u)
            # TVD/SSP RK2:
            # u1 = u^n + dt * L(u^n)
            # u^{n+1} = 0.5 * u^n + 0.5 * (u1 + dt * L(u1))
            u[i] = T(0.5) * u[i] + T(0.5) * (u1[i] + dt * rhs1[i] / V[idx[i]])
        end
        t += dt
    end

    return u
end

function _order_errors_1d_rk2(
    scheme;
    vel::Float64=1.0,
    t_final::Float64=0.2,
    cfl::Float64=0.5,
    ns=(32, 64, 128),
)
    errs_l2 = zeros(Float64, length(ns))
    errs_l1 = zeros(Float64, length(ns))
    @inbounds for k in eachindex(ns)
        sys = build_periodic_1d_system(;
            nx=ns[k],
            scheme=scheme,
            vel=vel,
            kappa=0.0,
        )
        u0 = _sine_state(sys, 0.0, vel)
        u_num = _rk2_ssp_integrate(sys, u0, t_final; cfl=cfl)
        u_exact = _sine_state(sys, t_final, vel)
        errs_l2[k] = _l2_weighted_error(sys, u_num, u_exact)
        errs_l1[k] = _l1_weighted_error(sys, u_num, u_exact)
    end

    p1_l2 = log2(errs_l2[1] / errs_l2[2])
    p2_l2 = log2(errs_l2[2] / errs_l2[3])
    p1_l1 = log2(errs_l1[1] / errs_l1[2])
    p2_l1 = log2(errs_l1[2] / errs_l1[3])
    return errs_l1, errs_l2, (p1_l1, p2_l1), (p1_l2, p2_l2)
end

@testset "Validation: smooth advection order (Centered + RK2)" begin
    errs_l1, errs_l2, orders_l1, orders_l2 = _order_errors_1d_rk2(CartesianOperators.Centered())
    @test all(isfinite, errs_l1)
    @test all(isfinite, errs_l2)
    @test errs_l1[3] < errs_l1[2] < errs_l1[1]
    @test errs_l2[3] < errs_l2[2] < errs_l2[1]
    @test orders_l1[1] > 1.8
    @test orders_l1[2] > 1.8
    @test orders_l2[1] > 1.8
    @test orders_l2[2] > 1.8
end

@testset "Validation: smooth advection order (MUSCL-MC + RK2)" begin
    errs_l1, errs_l2, orders_l1, orders_l2 = _order_errors_1d_rk2(CartesianOperators.MUSCL(CartesianOperators.MC()))
    @test all(isfinite, errs_l1)
    @test all(isfinite, errs_l2)
    @test errs_l1[3] < errs_l1[2] < errs_l1[1]
    @test errs_l2[3] < errs_l2[2] < errs_l2[1]
    # L1 is the primary FV convergence metric for limiter schemes.
    @test orders_l1[1] > 1.8
    @test orders_l1[2] > 1.8
    # L2 stays lower due limiter activation near smooth extrema.
    @test orders_l2[1] > 1.5
    @test orders_l2[2] > 1.5
end

@testset "Validation: smooth advection order (MUSCL-VanLeer + RK2)" begin
    errs_l1, errs_l2, orders_l1, orders_l2 = _order_errors_1d_rk2(CartesianOperators.MUSCL(CartesianOperators.VanLeer()))
    @test all(isfinite, errs_l1)
    @test all(isfinite, errs_l2)
    @test errs_l1[3] < errs_l1[2] < errs_l1[1]
    @test errs_l2[3] < errs_l2[2] < errs_l2[1]
    @test orders_l1[1] > 1.8
    @test orders_l1[2] > 1.8
    @test orders_l2[1] > 1.5
    @test orders_l2[2] > 1.5
end
