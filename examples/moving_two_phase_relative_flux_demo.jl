using CartesianGeometry: geometric_moments, nan
using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinSolverCore: LinearSystem
using PenguinTransport
using SparseArrays: spzeros

function main()
    grid_space = (range(0.0, 1.0; length=65),)
    body_space(x) = x - 0.53
    cap_ref = assembled_capacity(geometric_moments(body_space, grid_space, Float64, nan; method=:vofijul); bc=0.0)
    nt = cap_ref.ntotal
    n1 = [cap_ref.n_γ[i][1] for i in 1:nt]

    grid = CartesianGrid((0.0,), (1.0,), (65,))
    body_time(x, t) = x - 0.53

    # Choose phase/interface velocities so relative signs differ from absolute signs.
    # Here: u1γ=0, u2γ=2n1, wγ=n1 => λ1<0 and λ2<0 on interface cells.
    z = zeros(nt)
    u1ω = (z,)
    u2ω = (z,)
    u1γ = (z,)
    u2γ = ([2.0 * n1[i] for i in 1:nt],)
    wγ = ([n1[i] for i in 1:nt],)

    model = MovingTransportModelTwoPhase(
        grid, body_time, u1ω, u1γ, u2ω, u2γ;
        wγ=wγ,
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Outflow(), right=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
        scheme=Centered(),
    )

    sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
    try
        assemble_unsteady_two_phase_moving!(sys, model, (zeros(nt), zeros(nt)), 0.0, 0.02, :BE)
        println("assembly succeeded (no both-inflow relative configuration detected)")
    catch ex
        println("assembly failed as expected for relative both-inflow configuration")
        println("  error: ", sprint(showerror, ex))
    end

    println("relative-flux demo complete")
end

main()
