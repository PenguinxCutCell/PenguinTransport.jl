using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

function periodic_1d_moments(nx::Int)
    x = collect(range(0.0, 1.0; length=nx + 1))
    full_domain(_x, _=0.0) = -1.0
    return CartesianGeometry.geometric_moments(full_domain, (x,), Float64, zero; method=:implicitintegration)
end

function build_periodic_system(nx::Int, scheme; vel=1.0)
    moments = periodic_1d_moments(nx)
    bc_adv = CartesianOperators.AdvBoxBC(
        (CartesianOperators.AdvPeriodic(Float64),),
        (CartesianOperators.AdvPeriodic(Float64),),
    )
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        bc_adv=bc_adv,
        scheme=scheme,
        vel_omega=vel,
        vel_gamma=vel,
    )
    return PenguinTransport.build_system(moments, prob)
end

function sine_state(sys, t, vel)
    idx = sys.dof_omega.indices
    x = sys.moments.xyz[1]
    u = zeros(Float64, length(idx))
    @inbounds for i in eachindex(idx)
        u[i] = sin(2pi * (x[idx[i]] + vel * t))
    end
    return u
end

function l2_weighted_error(sys, u_num, u_ref)
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

function l1_weighted_error(sys, u_num, u_ref)
    idx = sys.dof_omega.indices
    V = sys.moments.V
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(idx)
        w = V[idx[i]]
        e = abs(u_num[i] - u_ref[i])
        num += w * e
        den += w
    end
    return num / den
end

function integrate_rk2!(u, sys, t_final; cfl=0.5)
    idx = sys.dof_omega.indices
    V = sys.moments.V
    dt_cfl = PenguinTransport.cfl_dt(sys, u; cfl=cfl, include_diffusion=false)
    nsteps = max(1, ceil(Int, t_final / dt_cfl))
    dt = t_final / nsteps

    rhs0 = similar(u)
    rhs1 = similar(u)
    u1 = similar(u)
    t = 0.0
    @inbounds for _ in 1:nsteps
        PenguinSolverCore.rhs!(rhs0, sys, u, nothing, t)
        for i in eachindex(u)
            u1[i] = u[i] + dt * rhs0[i] / V[idx[i]]
        end

        PenguinSolverCore.rhs!(rhs1, sys, u1, nothing, t + dt)
        for i in eachindex(u)
            u[i] = 0.5 * u[i] + 0.5 * (u1[i] + dt * rhs1[i] / V[idx[i]])
        end
        t += dt
    end
    return u
end

function run_convergence(name, scheme; nx_list=(32, 64, 128, 256), vel=1.0, t_final=0.2, cfl=0.5)
    errs_l1 = Float64[]
    errs_l2 = Float64[]
    for nx in nx_list
        sys = build_periodic_system(nx, scheme; vel=vel)
        u0 = sine_state(sys, 0.0, vel)
        u = copy(u0)
        integrate_rk2!(u, sys, t_final; cfl=cfl)
        u_exact = sine_state(sys, t_final, vel)
        push!(errs_l1, l1_weighted_error(sys, u, u_exact))
        push!(errs_l2, l2_weighted_error(sys, u, u_exact))
    end

    orders_l1 = [log2(errs_l1[i] / errs_l1[i + 1]) for i in 1:(length(errs_l1) - 1)]
    orders_l2 = [log2(errs_l2[i] / errs_l2[i + 1]) for i in 1:(length(errs_l2) - 1)]
    println(name)
    for i in eachindex(nx_list)
        if i == 1
            println("  nx=$(nx_list[i])  L1=$(errs_l1[i])  L2=$(errs_l2[i])")
        else
            println("  nx=$(nx_list[i])  L1=$(errs_l1[i])  p1=$(orders_l1[i - 1])  L2=$(errs_l2[i])  p2=$(orders_l2[i - 1])")
        end
    end
    return (errs_l1, orders_l1), (errs_l2, orders_l2)
end

println("1D periodic sine advection (full domain level set = -1)")
println("time integrator: SSP RK2 (manual mass-matrix update), CFL=0.5")

(ctr_l1, ctr_l2) = run_convergence("Centered", CartesianOperators.Centered())
(mc_l1, mc_l2) = run_convergence("MUSCL + MC", CartesianOperators.MUSCL(CartesianOperators.MC()))
(vl_l1, vl_l2) = run_convergence("MUSCL + VanLeer", CartesianOperators.MUSCL(CartesianOperators.VanLeer()))

println()
println("Minimum observed order (L1 / L2):")
println("  Centered: $(minimum(ctr_l1[2])) / $(minimum(ctr_l2[2]))")
println("  MUSCL+MC: $(minimum(mc_l1[2])) / $(minimum(mc_l2[2]))")
println("  MUSCL+VanLeer: $(minimum(vl_l1[2])) / $(minimum(vl_l2[2]))")

if minimum(mc_l1[2]) < 1.8 || minimum(vl_l1[2]) < 1.8
    println()
    println("MUSCL is below clear second-order in L1 for this setup.")
    println("Check:")
    println("  - slope calculation")
    println("  - periodic boundary treatment")
    println("  - RK2 integration coupling (M^{-1} * rhs)")
    println("  - limiter implementation")
end
