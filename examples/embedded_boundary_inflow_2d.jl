using PenguinTransport
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using LinearAlgebra

function cut_moments(nx::Int=72, ny::Int=72)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    levelset(x, y, _=0.0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.22
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function build_sys(moments; embedded_inflow=nothing)
    vel = (0.9, 0.1)
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        scheme=CartesianOperators.Upwind1(),
        vel_omega=vel,
        vel_gamma=vel,
        embedded_inflow=embedded_inflow,
    )
    return PenguinTransport.build_system(moments, prob)
end

function count_inflow_faces(sys)
    n_in = 0
    n_out = 0
    tol = sqrt(eps(Float64))
    u1 = sys.uω_full[1]
    u2 = sys.uω_full[2]
    @inbounds for i in eachindex(sys.moments.interface_measure)
        if isfinite(sys.moments.interface_measure[i]) && sys.moments.interface_measure[i] > tol
            n = sys.moments.interface_normal[i]
            un = u1[i] * n[1] + u2[i] * n[2]
            if un < 0.0
                n_in += 1
            else
                n_out += 1
            end
        end
    end
    return n_in, n_out
end

function main()
    moms = cut_moments()
    sys_none = build_sys(moms; embedded_inflow=nothing)
    sys_in = build_sys(moms; embedded_inflow=1.0)

    u0 = zeros(Float64, length(sys_none.dof_omega.indices))
    du_none = similar(u0)
    du_in = similar(u0)

    PenguinSolverCore.rhs!(du_none, sys_none, u0, nothing, 0.0)
    PenguinSolverCore.rhs!(du_in, sys_in, u0, nothing, 0.0)

    nin, nout = count_inflow_faces(sys_in)

    println("2D embedded-boundary inflow closure")
    println("  Interface inflow cells  : ", nin)
    println("  Interface outflow cells : ", nout)
    println("  ||rhs|| without inflow BC: ", norm(du_none))
    println("  ||rhs|| with inflow BC   : ", norm(du_in))
    println("  ||difference||           : ", norm(du_in - du_none))
end

main()
