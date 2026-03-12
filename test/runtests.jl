using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PenguinTransport
using CartesianGeometry
using CartesianGrids
using CartesianOperators
using PenguinSolverCore
using PenguinBCs

function full_moments(grid)
    levelset(args...) = -1.0
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function circle_moments(grid; r=0.22, cx=0.5, cy=0.5)
    levelset(x, y, _=0) = sqrt((x - cx)^2 + (y - cy)^2) - r
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function circle_moments_complement(grid; r=0.22, cx=0.5, cy=0.5)
    levelset(x, y, _=0) = r - sqrt((x - cx)^2 + (y - cy)^2)
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function planar_moments_left(grid; x0=0.5)
    levelset(x, y=0, z=0) = x - x0
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function planar_moments_right(grid; x0=0.5)
    levelset(x, y=0, z=0) = x0 - x
    return geometric_moments(levelset, grid, Float64, nan; method=:vofijul)
end

function omega_view(model, state)
    return state[model.layout.offsets.ω]
end

function active_omega_mask(cap)
    N = length(cap.nnodes)
    LI = LinearIndices(cap.nnodes)
    mask = falses(cap.ntotal)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        halo = any(d -> I[d] == cap.nnodes[d], 1:N)
        halo && continue
        v = cap.buf.V[lin]
        mask[lin] = isfinite(v) && v > 0
    end
    return mask
end

function interface_mask(cap1, cap2)
    nt = cap1.ntotal
    mask = falses(nt)
    for i in 1:nt
        Γ1 = cap1.buf.Γ[i]
        Γ2 = cap2.buf.Γ[i]
        mask[i] = (isfinite(Γ1) && Γ1 > 0) || (isfinite(Γ2) && Γ2 > 0)
    end
    return mask
end

function interface_flux_metrics(model::TransportModelTwoPhase, x)
    cap1 = model.cap1
    cap2 = model.cap2
    lay = model.layout
    nt = cap1.ntotal
    N = length(cap1.nnodes)
    num_abs = 0.0
    den_abs = 0.0
    sum_signed = 0.0
    niface = 0
    for i in 1:nt
        Γ1 = cap1.buf.Γ[i]
        Γ2 = cap2.buf.Γ[i]
        Γ = 0.5 * (Γ1 + Γ2)
        (isfinite(Γ) && Γ > 0) || continue
        s1 = 0.0
        s2 = 0.0
        for d in 1:N
            s1 += model.u1γ[d][i] * cap1.n_γ[i][d]
            s2 += model.u2γ[d][i] * cap2.n_γ[i][d]
        end
        T1γ = x[lay.γ1[i]]
        T2γ = x[lay.γ2[i]]
        r = Γ * (s1 * T1γ + s2 * T2γ)
        sum_signed += r
        num_abs += abs(r)
        den_abs += abs(Γ * s1 * T1γ) + abs(Γ * s2 * T2γ)
        niface += 1
    end
    return (
        sum_signed=sum_signed,
        abs_sum=num_abs,
        rel_abs=num_abs / (den_abs + eps(Float64)),
        niface=niface,
    )
end

function _linf_error_region(cap, u_num, u_exact, pred)
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    err = 0.0
    used = 0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = cap.buf.V[lin]
        (isfinite(v) && v > 0) || continue
        x = cap.C_ω[lin][1]
        pred(x) || continue
        err = max(err, abs(u_num[lin] - u_exact))
        used += 1
    end
    return err, used
end

function _l2_weighted_error(cap, u_num, u_exact)
    V = cap.buf.V
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    num = 0.0
    den = 0.0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * (u_num[lin] - u_exact[lin])^2
        den += v
    end
    return sqrt(num / max(den, eps(Float64)))
end

@testset "API contract regressions" begin
    @testset "Scheme parsing and validation" begin
        grid_mono = (0.0:0.25:1.0,)
        cap_mono = assembled_capacity(full_moments(grid_mono); bc=0.0)
        nt_mono = cap_mono.ntotal
        bc_mono = BorderConditions(; left=Periodic(), right=Periodic())
        model_mono = TransportModelMono(cap_mono, (ones(nt_mono),), (ones(nt_mono),); bc_border=bc_mono, scheme=Centered())
        sys_mono = LinearSystem(spzeros(Float64, 2 * nt_mono, 2 * nt_mono), zeros(Float64, 2 * nt_mono))
        u0_mono = zeros(nt_mono)

        @test_nowarn assemble_unsteady_mono!(sys_mono, model_mono, u0_mono, 0.0, 0.1, :BE)
        @test_nowarn assemble_unsteady_mono!(sys_mono, model_mono, u0_mono, 0.0, 0.1, :CN)
        @test_nowarn assemble_unsteady_mono!(sys_mono, model_mono, u0_mono, 0.0, 0.1, 0.3)
        @test_throws ArgumentError assemble_unsteady_mono!(sys_mono, model_mono, u0_mono, 0.0, 0.1, :foo)
        @test_throws ArgumentError assemble_unsteady_mono!(sys_mono, model_mono, u0_mono, 0.0, 0.1, -0.1)
        @test_throws ArgumentError assemble_unsteady_mono!(sys_mono, model_mono, u0_mono, 0.0, 0.1, 1.2)
        @test_throws ArgumentError solve_unsteady!(model_mono, u0_mono, (0.0, 0.1); dt=0.1, scheme=:foo, save_history=false)

        grid_two = (0.0:0.25:1.0,)
        cap1 = assembled_capacity(planar_moments_left(grid_two; x0=0.5); bc=0.0)
        cap2 = assembled_capacity(planar_moments_right(grid_two; x0=0.5); bc=0.0)
        nt_two = cap1.ntotal
        z = zeros(nt_two)
        model_two = TransportModelTwoPhase(
            cap1, cap2,
            (z,), (z,),
            (z,), (z,);
            source1=0.0,
            source2=0.0,
            bc_border1=BorderConditions(; left=Outflow(), right=Outflow()),
            bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
            scheme=Centered(),
        )
        sys_two = LinearSystem(spzeros(Float64, 4 * nt_two, 4 * nt_two), zeros(Float64, 4 * nt_two))
        u0_two = (zeros(nt_two), zeros(nt_two))

        @test_nowarn assemble_unsteady_two_phase!(sys_two, model_two, u0_two, 0.0, 0.1, :BE)
        @test_nowarn assemble_unsteady_two_phase!(sys_two, model_two, u0_two, 0.0, 0.1, :CN)
        @test_nowarn assemble_unsteady_two_phase!(sys_two, model_two, u0_two, 0.0, 0.1, 0.7)
        @test_throws ArgumentError assemble_unsteady_two_phase!(sys_two, model_two, u0_two, 0.0, 0.1, :foo)
        @test_throws ArgumentError assemble_unsteady_two_phase!(sys_two, model_two, u0_two, 0.0, 0.1, -0.1)
        @test_throws ArgumentError assemble_unsteady_two_phase!(sys_two, model_two, u0_two, 0.0, 0.1, 1.2)
        @test_throws ArgumentError solve_unsteady!(model_two, u0_two, (0.0, 0.1); dt=0.1, scheme=1.2, save_history=false)
    end

    @testset "Direct assembly and solve wrapper consistency (:CN)" begin
        t0 = 0.0
        dt = 0.1

        grid_mono = (0.0:0.25:1.0,)
        cap_mono = assembled_capacity(full_moments(grid_mono); bc=0.0)
        nt_mono = cap_mono.ntotal
        bc_mono = BorderConditions(; left=Periodic(), right=Periodic())
        model_mono = TransportModelMono(cap_mono, (ones(nt_mono),), (ones(nt_mono),); bc_border=bc_mono, source=0.0, scheme=Centered())
        u0_mono = sin.(2π .* cap_mono.xyz[1])

        sys_dir_mono = LinearSystem(spzeros(Float64, 2 * nt_mono, 2 * nt_mono), zeros(Float64, 2 * nt_mono))
        assemble_unsteady_mono!(sys_dir_mono, model_mono, u0_mono, t0, dt, :CN)
        solve!(sys_dir_mono; method=:direct)

        res_mono = solve_unsteady!(model_mono, u0_mono, (t0, t0 + dt); dt=dt, scheme=:CN, save_history=true)
        mask_mono = active_omega_mask(cap_mono)
        @test norm(omega_view(model_mono, sys_dir_mono.x)[mask_mono] - omega_view(model_mono, res_mono.states[end])[mask_mono], Inf) < 1e-12

        grid_two = (0.0:0.25:1.0,)
        cap1 = assembled_capacity(planar_moments_left(grid_two; x0=0.5); bc=0.0)
        cap2 = assembled_capacity(planar_moments_right(grid_two; x0=0.5); bc=0.0)
        nt_two = cap1.ntotal
        z = zeros(nt_two)
        model_two = TransportModelTwoPhase(
            cap1, cap2,
            (z,), (z,),
            (z,), (z,);
            source1=0.0,
            source2=0.0,
            bc_border1=BorderConditions(; left=Outflow(), right=Outflow()),
            bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
            scheme=Centered(),
        )
        u0_two = (fill(0.2, nt_two), fill(-0.1, nt_two))

        sys_dir_two = LinearSystem(spzeros(Float64, 4 * nt_two, 4 * nt_two), zeros(Float64, 4 * nt_two))
        assemble_unsteady_two_phase!(sys_dir_two, model_two, u0_two, t0, dt, :CN)
        solve!(sys_dir_two; method=:direct)

        res_two = solve_unsteady!(model_two, u0_two, (t0, t0 + dt); dt=dt, scheme=:CN, save_history=true)
        mask1 = active_omega_mask(cap1)
        mask2 = active_omega_mask(cap2)
        @test norm(omega1_view(model_two, sys_dir_two.x)[mask1] - omega1_view(model_two, res_two.states[end])[mask1], Inf) < 1e-12
        @test norm(omega2_view(model_two, sys_dir_two.x)[mask2] - omega2_view(model_two, res_two.states[end])[mask2], Inf) < 1e-12
        @test norm(gamma1_view(model_two, sys_dir_two.x) - gamma1_view(model_two, res_two.states[end]), Inf) < 1e-12
        @test norm(gamma2_view(model_two, sys_dir_two.x) - gamma2_view(model_two, res_two.states[end]), Inf) < 1e-12
    end

    @testset "Mono embedded-interface closure: inflow/outflow/no-flow" begin
        grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
        cap = assembled_capacity(circle_moments(grid); bc=0.0)
        nt = cap.ntotal
        lay = layout_mono(nt).offsets
        zω = (zeros(nt), zeros(nt))
        g = 2.5

        # Sign-varying interface velocity (uγ·nγ changes sign around the circle).
        uγ = (ones(nt), zeros(nt))
        model = TransportModelMono(cap, zω, uγ; bc_interface=Inflow(g), scheme=Centered())
        sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
        assemble_steady_mono!(sys, model, 0.0)

        nin = 0
        nout = 0
        for i in 1:nt
            Γ = cap.buf.Γ[i]
            (isfinite(Γ) && Γ > 0) || continue
            s = uγ[1][i] * cap.n_γ[i][1] + uγ[2][i] * cap.n_γ[i][2]
            r = lay.γ[i]
            if s < 0
                nin += 1
                @test sys.A[r, lay.ω[i]] ≈ 0.0 atol=1e-12
                @test sys.A[r, lay.γ[i]] ≈ Γ atol=1e-12
                @test sys.b[r] ≈ Γ * g atol=1e-12
            else
                nout += 1
                @test sys.A[r, lay.ω[i]] ≈ -Γ atol=1e-12
                @test sys.A[r, lay.γ[i]] ≈ Γ atol=1e-12
                @test sys.b[r] ≈ 0.0 atol=1e-12
            end
        end
        @test nin > 0
        @test nout > 0

        # No-flow interface mode: recover continuity closure everywhere.
        zγ = (zeros(nt), zeros(nt))
        model0 = TransportModelMono(cap, zω, zγ; bc_interface=Inflow(g), scheme=Centered())
        sys0 = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
        assemble_steady_mono!(sys0, model0, 0.0)
        for i in 1:nt
            Γ = cap.buf.Γ[i]
            (isfinite(Γ) && Γ > 0) || continue
            r = lay.γ[i]
            @test sys0.A[r, lay.ω[i]] ≈ -Γ atol=1e-12
            @test sys0.A[r, lay.γ[i]] ≈ Γ atol=1e-12
            @test sys0.b[r] ≈ 0.0 atol=1e-12
        end
    end

    @testset "Two-phase both-inflow local configuration throws" begin
        grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
        cap1 = assembled_capacity(circle_moments(grid); bc=0.0)
        cap2 = assembled_capacity(circle_moments_complement(grid); bc=0.0)
        nt = cap1.ntotal
        z = zeros(nt)
        u1γ = ([-cap1.n_γ[i][1] for i in 1:nt], [-cap1.n_γ[i][2] for i in 1:nt])
        u2γ = ([-cap2.n_γ[i][1] for i in 1:nt], [-cap2.n_γ[i][2] for i in 1:nt])
        model = TransportModelTwoPhase(
            cap1, cap2,
            (z, z), u1γ,
            (z, z), u2γ;
            bc_border1=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
            bc_border2=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
            scheme=Centered(),
        )
        sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))

        err = try
            assemble_steady_two_phase!(sys, model, 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("both-inflow local configuration", sprint(showerror, err))
    end

    @testset "Sign-convention regression: positive inflow data gives negative steady state" begin
        grid = (0.0:0.05:1.0,)
        cap = assembled_capacity(full_moments(grid); bc=0.0)
        nt = cap.ntotal
        g = 1.3
        bc = BorderConditions(; left=Inflow(g), right=Outflow())
        model = TransportModelMono(cap, (ones(nt),), (ones(nt),); source=0.0, bc_border=bc, scheme=Upwind1())
        sys = solve_steady!(model)
        ω = omega_view(model, sys.x)
        mask = active_omega_mask(cap)

        # Current transport operator convention maps positive imposed inflow value g to -g in the solved scalar.
        @test maximum(abs.(ω[mask] .+ g)) < 1e-12
    end
end

@testset "Moving-geometry transport regressions" begin
    @testset "A. Moving mono frozen geometry matches fixed geometry" begin
        grid_space = (range(0.0, 1.0; length=65),)
        xγ = 0.53
        body_space(x) = x - xγ
        body_time(x, t) = x - xγ

        moms = geometric_moments(body_space, grid_space, Float64, nan; method=:vofijul)
        cap = assembled_capacity(moms; bc=0.0)
        nt = cap.ntotal
        uω = (ones(nt),)
        uγ = (ones(nt),)
        wγ = (zeros(nt),)
        bc = BorderConditions(; left=Inflow(0.0), right=Outflow())

        fixed = TransportModelMono(cap, uω, uγ; source=0.0, bc_border=bc, bc_interface=nothing, scheme=Upwind1())
        lay = fixed.layout.offsets
        nsys = last(lay.γ)
        u0 = sin.(2π .* cap.xyz[1])

        sys_fixed = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
        assemble_unsteady_mono!(sys_fixed, fixed, u0, 0.0, 0.02, :BE)
        solve!(sys_fixed; method=:direct, reuse_factorization=false)

        grid = CartesianGrid((0.0,), (1.0,), (65,))
        moving = MovingTransportModelMono(
            grid, body_time, uω, uγ;
            wγ=wγ,
            source=0.0,
            bc_border=bc,
            bc_interface=nothing,
            scheme=Upwind1(),
            geom_method=:vofijul,
        )
        sys_moving = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
        assemble_unsteady_mono_moving!(sys_moving, moving, u0, 0.0, 0.02, :BE)
        solve!(sys_moving; method=:direct, reuse_factorization=false)

        capm = moving.cap_slab
        @test !(capm === nothing)
        mask = active_omega_mask(capm)
        err = norm(sys_moving.x[lay.ω][mask] - sys_fixed.x[lay.ω][mask]) / max(norm(sys_fixed.x[lay.ω][mask]), 1e-14)
        @test err < 5e-4
    end

    @testset "B. Moving two-phase frozen geometry matches fixed geometry" begin
        grid_space = (range(0.0, 1.0; length=65),)
        xγ = 0.53
        body_space(x) = x - xγ
        body_time(x, t) = x - xγ
        moms1 = geometric_moments(body_space, grid_space, Float64, nan; method=:vofijul)
        moms2 = geometric_moments((x) -> -body_space(x), grid_space, Float64, nan; method=:vofijul)
        cap1 = assembled_capacity(moms1; bc=0.0)
        cap2 = assembled_capacity(moms2; bc=0.0)
        nt = cap1.ntotal

        u1ω = (ones(nt),)
        u1γ = (ones(nt),)
        u2ω = (2.0 .* ones(nt),)
        u2γ = (2.0 .* ones(nt),)
        wγ = (zeros(nt),)
        bc1 = BorderConditions(; left=Inflow(0.0), right=Outflow())
        bc2 = BorderConditions(; left=Outflow(), right=Outflow())

        fixed = TransportModelTwoPhase(
            cap1, cap2, u1ω, u1γ, u2ω, u2γ;
            source1=0.0,
            source2=0.0,
            bc_border1=bc1,
            bc_border2=bc2,
            scheme=Upwind1(),
        )
        lay = fixed.layout
        nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
        u0 = (sin.(2π .* cap1.xyz[1]), cos.(2π .* cap2.xyz[1]))

        sys_fixed = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
        assemble_unsteady_two_phase!(sys_fixed, fixed, u0, 0.0, 0.02, :BE)
        solve!(sys_fixed; method=:direct, reuse_factorization=false)

        grid = CartesianGrid((0.0,), (1.0,), (65,))
        moving = MovingTransportModelTwoPhase(
            grid, body_time, u1ω, u1γ, u2ω, u2γ;
            wγ=wγ,
            source1=0.0,
            source2=0.0,
            bc_border1=bc1,
            bc_border2=bc2,
            scheme=Upwind1(),
            geom_method=:vofijul,
        )
        sys_moving = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
        assemble_unsteady_two_phase_moving!(sys_moving, moving, u0, 0.0, 0.02, :BE)
        solve!(sys_moving; method=:direct, reuse_factorization=false)

        cap1m = moving.cap1_slab
        cap2m = moving.cap2_slab
        @test !(cap1m === nothing)
        @test !(cap2m === nothing)
        mask1 = active_omega_mask(cap1m)
        mask2 = active_omega_mask(cap2m)
        err1 = norm(sys_moving.x[lay.ω1][mask1] - sys_fixed.x[lay.ω1][mask1]) / max(norm(sys_fixed.x[lay.ω1][mask1]), 1e-14)
        err2 = norm(sys_moving.x[lay.ω2][mask2] - sys_fixed.x[lay.ω2][mask2]) / max(norm(sys_fixed.x[lay.ω2][mask2]), 1e-14)
        @test err1 < 2e-3
        @test err2 < 2e-3
    end

    @testset "C. Moving mono no-interface reduction" begin
        grid_space = (range(0.0, 1.0; length=65),)
        cap = assembled_capacity(full_moments(grid_space); bc=0.0)
        nt = cap.ntotal
        uω = (ones(nt),)
        uγ = (ones(nt),)
        bc = BorderConditions(; left=Periodic(), right=Periodic())
        u0 = sin.(2π .* cap.xyz[1])

        fixed = TransportModelMono(cap, uω, uγ; source=0.0, bc_border=bc, bc_interface=nothing, scheme=Upwind1())
        rf = solve_unsteady!(fixed, u0, (0.0, 0.05); dt=0.01, scheme=:CN, save_history=false)
        ωf = omega_view(fixed, rf.states[end])

        grid = CartesianGrid((0.0,), (1.0,), (65,))
        body_full(x, t) = -1.0
        moving = MovingTransportModelMono(
            grid, body_full, uω, uγ;
            wγ=(zeros(nt),),
            source=0.0,
            bc_border=bc,
            bc_interface=nothing,
            scheme=Upwind1(),
            geom_method=:vofijul,
        )
        rm = solve_unsteady_moving!(moving, u0, (0.0, 0.05); dt=0.01, scheme=:CN, save_history=false)
        capm = moving.cap_slab
        @test !(capm === nothing)
        mask = active_omega_mask(capm)
        ωm = rm.states[end][moving.layout.offsets.ω]
        @test norm(ωf[mask] - ωm[mask], Inf) < 1e-10
    end

    @testset "D. Relative-speed logic (uγ = wγ => λ = 0)" begin
        grid = CartesianGrid((0.0,), (1.0,), (65,))
        body(x, t) = x - 0.53
        nt = prod(grid.n)
        z = zeros(nt)
        uγ = (ones(nt),)
        wγ = (ones(nt),)
        model = MovingTransportModelMono(
            grid, body, (z,), uγ;
            wγ=wγ,
            source=0.0,
            bc_border=BorderConditions(; left=Outflow(), right=Outflow()),
            bc_interface=Inflow(3.0),
            scheme=Centered(),
            geom_method=:vofijul,
        )
        sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
        assemble_unsteady_mono_moving!(sys, model, zeros(nt), 0.0, 0.02, :BE)

        cap = model.cap_slab
        @test !(cap === nothing)
        lay = model.layout.offsets
        checked = 0
        for i in 1:nt
            Γ = cap.buf.Γ[i]
            (isfinite(Γ) && Γ > 0) || continue
            r = lay.γ[i]
            @test sys.A[r, lay.ω[i]] ≈ -Γ atol=1e-12
            @test sys.A[r, lay.γ[i]] ≈ Γ atol=1e-12
            @test sys.b[r] ≈ 0.0 atol=1e-12
            checked += 1
        end
        @test checked > 0
    end

    @testset "E. Moving mono constant-state geometric sanity" begin
        grid = CartesianGrid((0.0,), (1.0,), (81,))
        U = 0.4
        body(x, t) = abs(x - (0.45 + U * t)) - 0.2
        nt = prod(grid.n)
        vel = (fill(U, nt),)
        cst = 1.7
        model = MovingTransportModelMono(
            grid, body, vel, vel;
            wγ=vel,
            source=0.0,
            bc_border=BorderConditions(; left=Periodic(), right=Periodic()),
            bc_interface=nothing,
            scheme=Upwind1(),
            geom_method=:vofijul,
        )
        res = solve_unsteady_moving!(model, fill(cst, nt), (0.0, 0.1); dt=0.02, scheme=:BE, save_history=false)
        cap = model.cap_slab
        @test !(cap === nothing)
        mask = active_omega_mask(cap)
        ω = res.states[end][model.layout.offsets.ω]
        @test all(isfinite, ω[mask])
        m0 = sum(model.Vn[i] * cst for i in eachindex(model.Vn) if mask[i])
        m1 = sum(model.Vn1[i] * ω[i] for i in eachindex(model.Vn1) if mask[i])
        rel_mass = abs(m1 - m0) / max(abs(m0), 1e-14)
        @test rel_mass < 5e-2
    end

    @testset "F. Moving mono mesh convergence (Upwind1)" begin
        function run_moving_mono_err(n)
            grid = CartesianGrid((0.0,), (1.0,), (n,))
            nt = prod(grid.n)
            U = 0.4
            tend = 0.1
            body(x, t) = abs(x - (0.35 + U * t)) - 0.2
            vel = (fill(U, nt),)
            model = MovingTransportModelMono(
                grid, body, vel, vel;
                wγ=vel,
                source=0.0,
                bc_border=BorderConditions(; left=Periodic(), right=Periodic()),
                bc_interface=nothing,
                scheme=Upwind1(),
                geom_method=:vofijul,
            )
            x = collect(range(0.0, 1.0; length=n))
            u0 = sin.(2π .* x)
            dt = 0.4 * (1.0 / (n - 1)) / U
            res = solve_unsteady_moving!(model, u0, (0.0, tend); dt=dt, scheme=:BE, save_history=false)
            cap = model.cap_slab
            @test !(cap === nothing)
            ω = res.states[end][model.layout.offsets.ω]
            exact = [sin(2π * (cap.C_ω[i][1] - U * tend)) for i in 1:nt]
            return _l2_weighted_error(cap, ω, exact)
        end

        errs = [run_moving_mono_err(n) for n in (33, 65, 129)]
        ord1 = log(errs[1] / errs[2]) / log(2)
        ord2 = log(errs[2] / errs[3]) / log(2)
        @test errs[3] < errs[2] < errs[1]
        @test min(ord1, ord2) >= 0.15
    end

    @testset "G. Moving two-phase mesh convergence (Upwind1)" begin
        function run_moving_two_err(n)
            grid = CartesianGrid((0.0,), (1.0,), (n,))
            nt = prod(grid.n)
            c = 0.4
            a0 = 0.47
            tend = 0.1
            body1(x, t) = x - (a0 + c * t)
            vel = (fill(c, nt),)
            bc = BorderConditions(; left=Periodic(), right=Periodic())
            model = MovingTransportModelTwoPhase(
                grid, body1, vel, vel, vel, vel;
                wγ=vel,
                source1=0.0,
                source2=0.0,
                bc_border1=bc,
                bc_border2=bc,
                scheme=Upwind1(),
                geom_method=:vofijul,
            )
            x = collect(range(0.0, 1.0; length=n))
            u01 = sin.(2π .* x)
            u02 = cos.(2π .* x)
            dt = 0.4 * (1.0 / (n - 1)) / c
            res = solve_unsteady_moving!(model, (u01, u02), (0.0, tend); dt=dt, scheme=:BE, save_history=false)
            cap1 = model.cap1_slab
            cap2 = model.cap2_slab
            @test !(cap1 === nothing)
            @test !(cap2 === nothing)
            lay = model.layout
            ω1 = res.states[end][lay.ω1]
            ω2 = res.states[end][lay.ω2]
            ex1 = [sin(2π * (cap1.C_ω[i][1] - c * tend)) for i in 1:nt]
            ex2 = [cos(2π * (cap2.C_ω[i][1] - c * tend)) for i in 1:nt]
            return _l2_weighted_error(cap1, ω1, ex1), _l2_weighted_error(cap2, ω2, ex2)
        end

        errs = [run_moving_two_err(n) for n in (33, 65, 129)]
        e1 = [e[1] for e in errs]
        e2 = [e[2] for e in errs]
        o11 = log(e1[1] / e1[2]) / log(2)
        o12 = log(e1[2] / e1[3]) / log(2)
        o21 = log(e2[1] / e2[2]) / log(2)
        o22 = log(e2[2] / e2[3]) / log(2)
        @test e1[3] < e1[2] < e1[1]
        @test e2[3] < e2[2] < e2[1]
        @test min(o11, o12) >= 0.15
        @test min(o21, o22) >= 0.15
    end

    @testset "H. Moving two-phase both-inflow rejection uses relative speeds" begin
        grid_space = (range(0.0, 1.0; length=65),)
        body_space(x) = x - 0.53
        moms1 = geometric_moments(body_space, grid_space, Float64, nan; method=:vofijul)
        cap1_ref = assembled_capacity(moms1; bc=0.0)
        nt = cap1_ref.ntotal
        n1 = [cap1_ref.n_γ[i][1] for i in 1:nt]

        grid = CartesianGrid((0.0,), (1.0,), (65,))
        body_time(x, t) = x - 0.53
        z = zeros(nt)
        u1γ = (z,)
        u2γ = ([2.0 * n1[i] for i in 1:nt],)
        wγ = ([n1[i] for i in 1:nt],)

        model = MovingTransportModelTwoPhase(
            grid, body_time, (z,), u1γ, (z,), u2γ;
            wγ=wγ,
            source1=0.0,
            source2=0.0,
            bc_border1=BorderConditions(; left=Outflow(), right=Outflow()),
            bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
            scheme=Centered(),
            geom_method=:vofijul,
        )
        sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
        err = try
            assemble_unsteady_two_phase_moving!(sys, model, (zeros(nt), zeros(nt)), 0.0, 0.02, :BE)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("relative speeds", sprint(showerror, err))
    end

    @testset "I. Moving smooth periodic convergence (> 1.0)" begin
        function run_moving_mono_high_order(n)
            grid = CartesianGrid((0.0,), (1.0,), (n,))
            nt = prod(grid.n)
            body_full(x, t) = -1.0
            vel = (ones(nt),)
            model = MovingTransportModelMono(
                grid, body_full, vel, vel;
                wγ=(zeros(nt),),
                source=0.0,
                bc_border=BorderConditions(; left=Periodic(), right=Periodic()),
                bc_interface=nothing,
                scheme=Centered(),
                geom_method=:vofijul,
            )

            x = collect(range(0.0, 1.0; length=n))
            u0 = sin.(2π .* x)
            dt = 0.5 * (1.0 / (n - 1))
            res = solve_unsteady_moving!(model, u0, (0.0, 1.0); dt=dt, scheme=:CN, save_history=false)
            cap = model.cap_slab
            @test !(cap === nothing)
            ω = res.states[end][model.layout.offsets.ω]
            exact = sin.(2π .* x) # one full period on the same periodic grid points
            return _l2_weighted_error(cap, ω, exact)
        end

        function run_moving_two_high_order(n)
            grid = CartesianGrid((0.0,), (1.0,), (n,))
            nt = prod(grid.n)
            body_full(x, t) = -1.0
            vel = (ones(nt),)
            bc = BorderConditions(; left=Periodic(), right=Periodic())
            model = MovingTransportModelTwoPhase(
                grid, body_full, vel, vel, vel, vel;
                body2=body_full,
                wγ=(zeros(nt),),
                source1=0.0,
                source2=0.0,
                bc_border1=bc,
                bc_border2=bc,
                scheme=Centered(),
                geom_method=:vofijul,
            )

            x = collect(range(0.0, 1.0; length=n))
            u01 = sin.(2π .* x)
            u02 = cos.(2π .* x)
            dt = 0.5 * (1.0 / (n - 1))
            res = solve_unsteady_moving!(model, (u01, u02), (0.0, 1.0); dt=dt, scheme=:CN, save_history=false)
            cap1 = model.cap1_slab
            cap2 = model.cap2_slab
            @test !(cap1 === nothing)
            @test !(cap2 === nothing)
            lay = model.layout
            ω1 = res.states[end][lay.ω1]
            ω2 = res.states[end][lay.ω2]
            ex1 = sin.(2π .* x)
            ex2 = cos.(2π .* x)
            return _l2_weighted_error(cap1, ω1, ex1), _l2_weighted_error(cap2, ω2, ex2)
        end

        em = [run_moving_mono_high_order(n) for n in (33, 65, 129)]
        om1 = log(em[1] / em[2]) / log(2)
        om2 = log(em[2] / em[3]) / log(2)
        @test em[3] < em[2] < em[1]
        @test min(om1, om2) > 1.0

        et = [run_moving_two_high_order(n) for n in (33, 65, 129)]
        e1 = [e[1] for e in et]
        e2 = [e[2] for e in et]
        o11 = log(e1[1] / e1[2]) / log(2)
        o12 = log(e1[2] / e1[3]) / log(2)
        o21 = log(e2[1] / e2[2]) / log(2)
        o22 = log(e2[2] / e2[3]) / log(2)
        @test e1[3] < e1[2] < e1[1]
        @test e2[3] < e2[2] < e2[1]
        @test min(o11, o12) > 1.0
        @test min(o21, o22) > 1.0
    end
end

@testset "1D periodic advection schemes" begin
    grid = (0.0:0.05:1.0,)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal
    x = cap.xyz[1]
    u0 = sin.(2π .* x)

    uω = (ones(nt),)
    uγ = (ones(nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())

    m_up = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Upwind1())
    r_up = solve_unsteady!(m_up, u0, (0.0, 1.0); dt=0.02, scheme=:BE, save_history=false)
    uf_up = omega_view(m_up, r_up.states[end])

    @test all(isfinite, uf_up)
    @test norm(uf_up, Inf) > 0

    m_c = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=Centered())
    r_c = solve_unsteady!(m_c, u0, (0.0, 1.0); dt=0.02, scheme=:BE, save_history=false)
    uf_c = omega_view(m_c, r_c.states[end])
    @test all(isfinite, uf_c)
end

@testset "Box inflow/outflow switching" begin
    grid = (0.0:0.2:1.0, 0.0:0.2:1.0)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal

    bc = BorderConditions(
        ;
        left=Inflow(1.0),
        right=Inflow(0.0),
        bottom=Outflow(),
        top=Outflow(),
    )

    model_pos = TransportModelMono(cap, (ones(nt), zeros(nt)), (ones(nt), zeros(nt)); bc_border=bc, scheme=Upwind1())
    res_pos = solve_unsteady!(model_pos, zeros(nt), (0.0, 0.6); dt=0.05, scheme=:BE, save_history=false)
    u_pos = omega_view(model_pos, res_pos.states[end])

    model_neg = TransportModelMono(cap, (-ones(nt), zeros(nt)), (-ones(nt), zeros(nt)); bc_border=bc, scheme=Upwind1())
    res_neg = solve_unsteady!(model_neg, zeros(nt), (0.0, 0.6); dt=0.05, scheme=:BE, save_history=false)
    u_neg = omega_view(model_neg, res_neg.states[end])

    @test norm(u_pos - u_neg, 2) > 1e-3
end

@testset "Embedded interface no-flow closure" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap = assembled_capacity(circle_moments(grid); bc=0.0)
    nt = cap.ntotal
    lay = layout_mono(nt).offsets

    zγ = (zeros(nt), zeros(nt))
    zω = (zeros(nt), zeros(nt))

    model_a = TransportModelMono(cap, zω, zγ; bc_interface=2.5)
    sys_a = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_steady_mono!(sys_a, model_a, 0.0)

    model_b = TransportModelMono(cap, zω, zγ; bc_interface=-3.0)
    sys_b = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_steady_mono!(sys_b, model_b, 0.0)

    iface = findall(i -> cap.buf.Γ[i] > 0, 1:nt)
    @test !isempty(iface)
    for i in iface
        r = lay.γ[i]
        γ = cap.buf.Γ[i]
        @test sys_a.A[r, lay.γ[i]] ≈ γ atol=1e-12
        @test sys_a.A[r, lay.ω[i]] ≈ -γ atol=1e-12
        @test sys_a.b[r] ≈ 0.0 atol=1e-12

        @test sys_b.A[r, lay.γ[i]] ≈ γ atol=1e-12
        @test sys_b.A[r, lay.ω[i]] ≈ -γ atol=1e-12
        @test sys_b.b[r] ≈ 0.0 atol=1e-12
    end

    @test norm(sys_a.A - sys_b.A) ≤ 1e-12
    @test norm(sys_a.b - sys_b.b) ≤ 1e-12
end

@testset "Embedded interface sign-based inflow/outflow BC" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap = assembled_capacity(circle_moments(grid); bc=0.0)
    nt = cap.ntotal
    lay = layout_mono(nt).offsets

    zω = (zeros(nt), zeros(nt))
    uγ = (ones(nt), zeros(nt))
    g = 2.5

    model = TransportModelMono(cap, zω, uγ; bc_interface=Inflow(g), scheme=Centered())
    sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_steady_mono!(sys, model, 0.0)

    n_in = 0
    n_out = 0
    for i in 1:nt
        Γ = cap.buf.Γ[i]
        (isfinite(Γ) && Γ > 0) || continue
        s = uγ[1][i] * cap.n_γ[i][1] + uγ[2][i] * cap.n_γ[i][2]
        r = lay.γ[i]
        if s < 0
            n_in += 1
            @test sys.A[r, lay.ω[i]] ≈ 0.0 atol=1e-12
            @test sys.A[r, lay.γ[i]] ≈ Γ atol=1e-12
            @test sys.b[r] ≈ Γ * g atol=1e-12
        else
            n_out += 1
            @test sys.A[r, lay.ω[i]] ≈ -Γ atol=1e-12
            @test sys.A[r, lay.γ[i]] ≈ Γ atol=1e-12
            @test sys.b[r] ≈ 0.0 atol=1e-12
        end
    end

    @test n_in > 0
    @test n_out > 0
end

@testset "Embedded interface no inflow value => continuity closure" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap = assembled_capacity(circle_moments(grid); bc=0.0)
    nt = cap.ntotal
    lay = layout_mono(nt).offsets

    zω = (zeros(nt), zeros(nt))
    uγ = (ones(nt), zeros(nt))
    model = TransportModelMono(cap, zω, uγ; bc_interface=nothing, scheme=Centered())
    sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_steady_mono!(sys, model, 0.0)

    n_iface = 0
    for i in 1:nt
        Γ = cap.buf.Γ[i]
        (isfinite(Γ) && Γ > 0) || continue
        n_iface += 1
        r = lay.γ[i]
        @test sys.A[r, lay.ω[i]] ≈ -Γ atol=1e-12
        @test sys.A[r, lay.γ[i]] ≈ Γ atol=1e-12
        @test sys.b[r] ≈ 0.0 atol=1e-12
    end

    @test n_iface > 0
end

@testset "Time-dependent embedded inflow disables constant reuse" begin
    grid = (0.0:0.2:1.0, 0.0:0.2:1.0)
    cap = assembled_capacity(circle_moments(grid); bc=0.0)
    nt = cap.ntotal

    uω = (zeros(nt), zeros(nt))
    uγ = (ones(nt), zeros(nt))
    bc = BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow())
    model = TransportModelMono(cap, uω, uγ; bc_border=bc, bc_interface=Inflow((x, y, t) -> 1 + t), scheme=Centered())
    res = solve_unsteady!(model, zeros(nt), (0.0, 0.2); dt=0.1, scheme=:BE, save_history=false)
    @test res.reused_constant_operator == false
end

@testset "Two-phase interface row pattern: one inflow / one outflow" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap1 = assembled_capacity(circle_moments(grid); bc=0.0)
    cap2 = assembled_capacity(circle_moments_complement(grid); bc=0.0)
    nt = cap1.ntotal

    z = zeros(nt)
    u1ω = (z, z)
    u2ω = (z, z)
    # Force phase 1 inflow (s1 < 0) and phase 2 outflow (s2 > 0) on interface.
    u1γ = ([-cap1.n_γ[i][1] for i in 1:nt], [-cap1.n_γ[i][2] for i in 1:nt])
    u2γ = ([ cap2.n_γ[i][1] for i in 1:nt], [ cap2.n_γ[i][2] for i in 1:nt])

    model = TransportModelTwoPhase(
        cap1, cap2, u1ω, u1γ, u2ω, u2γ;
        bc_border1=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
        scheme=Centered(),
    )

    sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
    assemble_steady_two_phase!(sys, model, 0.0)
    lay = model.layout

    checked = 0
    for i in 1:nt
        Γ1 = cap1.buf.Γ[i]
        Γ2 = cap2.buf.Γ[i]
        Γ = 0.5 * (Γ1 + Γ2)
        (isfinite(Γ) && Γ > 0) || continue

        s1 = u1γ[1][i] * cap1.n_γ[i][1] + u1γ[2][i] * cap1.n_γ[i][2]
        s2 = u2γ[1][i] * cap2.n_γ[i][1] + u2γ[2][i] * cap2.n_γ[i][2]
        @test s1 < 0
        @test s2 > 0

        r_flux = lay.γ1[i]
        @test sys.A[r_flux, lay.γ1[i]] ≈ Γ * s1 atol=1e-12
        @test sys.A[r_flux, lay.γ2[i]] ≈ Γ * s2 atol=1e-12
        @test sys.A[r_flux, lay.ω1[i]] ≈ 0.0 atol=1e-12
        @test sys.A[r_flux, lay.ω2[i]] ≈ 0.0 atol=1e-12
        @test sys.b[r_flux] ≈ 0.0 atol=1e-12

        r_clo = lay.γ2[i]
        @test sys.A[r_clo, lay.ω2[i]] ≈ -Γ atol=1e-12
        @test sys.A[r_clo, lay.γ2[i]] ≈ Γ atol=1e-12
        @test sys.A[r_clo, lay.ω1[i]] ≈ 0.0 atol=1e-12
        @test sys.A[r_clo, lay.γ1[i]] ≈ 0.0 atol=1e-12
        @test sys.b[r_clo] ≈ 0.0 atol=1e-12
        checked += 1
    end
    @test checked > 0
end

@testset "Two-phase interface both-inflow throws" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap1 = assembled_capacity(circle_moments(grid); bc=0.0)
    cap2 = assembled_capacity(circle_moments_complement(grid); bc=0.0)
    nt = cap1.ntotal

    z = zeros(nt)
    u1ω = (z, z)
    u2ω = (z, z)
    u1γ = ([-cap1.n_γ[i][1] for i in 1:nt], [-cap1.n_γ[i][2] for i in 1:nt])
    u2γ = ([-cap2.n_γ[i][1] for i in 1:nt], [-cap2.n_γ[i][2] for i in 1:nt])

    model = TransportModelTwoPhase(
        cap1, cap2, u1ω, u1γ, u2ω, u2γ;
        bc_border1=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
        scheme=Centered(),
    )
    sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
    @test_throws ArgumentError assemble_steady_two_phase!(sys, model, 0.0)
end

@testset "Two-phase constant-state interface residual is zero" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap1 = assembled_capacity(circle_moments(grid); bc=0.0)
    cap2 = assembled_capacity(circle_moments_complement(grid); bc=0.0)
    nt = cap1.ntotal

    z = zeros(nt)
    u1ω = (z, z)
    u2ω = (z, z)
    u1γ = (ones(nt), zeros(nt))
    u2γ = (ones(nt), zeros(nt))

    model = TransportModelTwoPhase(
        cap1, cap2, u1ω, u1γ, u2ω, u2γ;
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
        scheme=Centered(),
    )
    sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
    assemble_steady_two_phase!(sys, model, 0.0)

    c0 = 1.7
    xconst = fill(c0, 4 * nt)
    r = sys.A * xconst - sys.b
    lay = model.layout

    iface = findall(i -> cap1.buf.Γ[i] > 0, 1:nt)
    @test !isempty(iface)
    @test maximum(abs.(r[lay.γ1][iface])) < 1e-12
    @test maximum(abs.(r[lay.γ2][iface])) < 1e-12
end

@testset "Two-phase unsteady constant operator reuse" begin
    grid = (0.0:0.1:1.0,)
    cap1 = assembled_capacity(planar_moments_left(grid); bc=0.0)
    cap2 = assembled_capacity(planar_moments_right(grid); bc=0.0)
    nt = cap1.ntotal

    model = TransportModelTwoPhase(
        cap1, cap2,
        (ones(nt),), (ones(nt),),
        (ones(nt),), (ones(nt),);
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Inflow(1.0), right=Outflow()),
        bc_border2=BorderConditions(; left=Inflow(1.0), right=Outflow()),
        scheme=Upwind1(),
    )

    res = solve_unsteady!(model, (ones(nt), ones(nt)), (0.0, 0.2); dt=0.05, scheme=:BE, save_history=false)
    @test res.reused_constant_operator == true
end

@testset "Two-phase interface local conservation (steady solve)" begin
    grid = (0.0:0.05:1.0,)
    cap1 = assembled_capacity(planar_moments_left(grid; x0=0.47); bc=0.0)
    cap2 = assembled_capacity(planar_moments_right(grid; x0=0.47); bc=0.0)
    nt = cap1.ntotal

    model = TransportModelTwoPhase(
        cap1, cap2,
        (ones(nt),), (ones(nt),),
        (2.0 .* ones(nt),), (2.0 .* ones(nt),);
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Inflow(1.0), right=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
        scheme=Upwind1(),
    )
    sys = solve_steady!(model)
    met = interface_flux_metrics(model, sys.x)
    @test met.niface > 0
    @test abs(met.sum_signed) < 1e-12
    @test met.rel_abs < 1e-12
end

@testset "Two-phase 1D planar zero-source interface ratio" begin
    x0 = 0.47
    dx = 0.05
    grid = (0.0:dx:1.0,)
    cap1 = assembled_capacity(planar_moments_left(grid; x0=x0); bc=0.0)
    cap2 = assembled_capacity(planar_moments_right(grid; x0=x0); bc=0.0)
    nt = cap1.ntotal
    g = 1.3

    model = TransportModelTwoPhase(
        cap1, cap2,
        (ones(nt),), (ones(nt),),
        (2.0 .* ones(nt),), (2.0 .* ones(nt),);
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
        scheme=Upwind1(),
    )
    sys = solve_steady!(model)
    lay = model.layout
    iface = findall(interface_mask(cap1, cap2))
    @test !isempty(iface)
    for i in iface
        s1 = model.u1γ[1][i] * cap1.n_γ[i][1]
        s2 = model.u2γ[1][i] * cap2.n_γ[i][1]
        T1γ = sys.x[lay.γ1[i]]
        T2γ = sys.x[lay.γ2[i]]
        @test abs(s1 * T1γ + s2 * T2γ) < 1e-12
        @test abs(T2γ - (-(s1 / s2)) * T1γ) < 1e-12
    end

    # Upstream phase-1 region should remain constant (solver sign convention gives -g).
    ω1 = sys.x[lay.ω1]
    e_up, n_up = _linf_error_region(cap1, ω1, -g, x -> x <= x0 - 2dx)
    @test n_up > 0
    @test e_up < 1e-12
end

@testset "Two-phase 1D planar source in phase 1 transfers interface flux" begin
    x0 = 0.47
    grid = (0.0:0.025:1.0,)
    cap1 = assembled_capacity(planar_moments_left(grid; x0=x0); bc=0.0)
    cap2 = assembled_capacity(planar_moments_right(grid; x0=x0); bc=0.0)
    nt = cap1.ntotal
    g = 1.3
    σ = 0.4

    model0 = TransportModelTwoPhase(
        cap1, cap2,
        (ones(nt),), (ones(nt),),
        (2.0 .* ones(nt),), (2.0 .* ones(nt),);
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
        scheme=Upwind1(),
    )
    modelσ = TransportModelTwoPhase(
        cap1, cap2,
        (ones(nt),), (ones(nt),),
        (2.0 .* ones(nt),), (2.0 .* ones(nt),);
        source1=σ,
        source2=0.0,
        bc_border1=BorderConditions(; left=Inflow(g), right=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
        scheme=Upwind1(),
    )

    sys0 = solve_steady!(model0)
    sysσ = solve_steady!(modelσ)
    lay = model0.layout
    iface = findall(interface_mask(cap1, cap2))
    @test length(iface) == 1
    i = iface[1]

    met0 = interface_flux_metrics(model0, sys0.x)
    metσ = interface_flux_metrics(modelσ, sysσ.x)
    @test met0.rel_abs < 1e-12
    @test metσ.rel_abs < 1e-12

    # With positive source in phase 1, interface values shift consistently.
    @test sysσ.x[lay.γ1[i]] > sys0.x[lay.γ1[i]]
    @test sysσ.x[lay.γ2[i]] > sys0.x[lay.γ2[i]]
end

@testset "Two-phase reversed-flow row selection is not phase-biased" begin
    grid = (0.0:0.05:1.0,)
    cap1 = assembled_capacity(planar_moments_left(grid; x0=0.47); bc=0.0)
    cap2 = assembled_capacity(planar_moments_right(grid; x0=0.47); bc=0.0)
    nt = cap1.ntotal

    z = zeros(nt)
    model = TransportModelTwoPhase(
        cap1, cap2,
        (z,), (-ones(nt),),  # phase 1 inflow at interface
        (z,), (-2.0 .* ones(nt),);  # phase 2 outflow at interface
        bc_border1=BorderConditions(; left=Outflow(), right=Outflow()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow()),
        scheme=Centered(),
    )

    sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
    assemble_steady_two_phase!(sys, model, 0.0)
    lay = model.layout

    iface = findall(interface_mask(cap1, cap2))
    @test !isempty(iface)
    for i in iface
        Γ = 0.5 * (cap1.buf.Γ[i] + cap2.buf.Γ[i])
        s1 = model.u1γ[1][i] * cap1.n_γ[i][1]
        s2 = model.u2γ[1][i] * cap2.n_γ[i][1]
        @test s1 < 0
        @test s2 > 0
        # Reversed flow branch should place flux equation on γ1 and closure on γ2.
        r1 = lay.γ1[i]
        @test sys.A[r1, lay.γ1[i]] ≈ Γ * s1 atol=1e-12
        @test sys.A[r1, lay.γ2[i]] ≈ Γ * s2 atol=1e-12
        @test sys.A[r1, lay.ω1[i]] ≈ 0.0 atol=1e-12
        @test sys.A[r1, lay.ω2[i]] ≈ 0.0 atol=1e-12

        r2 = lay.γ2[i]
        @test sys.A[r2, lay.ω2[i]] ≈ -Γ atol=1e-12
        @test sys.A[r2, lay.γ2[i]] ≈ Γ atol=1e-12
        @test sys.A[r2, lay.ω1[i]] ≈ 0.0 atol=1e-12
        @test sys.A[r2, lay.γ1[i]] ≈ 0.0 atol=1e-12
    end
end

@testset "Two-phase 2D planar transport sanity + interface conservation" begin
    x0 = 0.47
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap1 = assembled_capacity(planar_moments_left(grid; x0=x0); bc=0.0)
    cap2 = assembled_capacity(planar_moments_right(grid; x0=x0); bc=0.0)
    nt = cap1.ntotal

    model = TransportModelTwoPhase(
        cap1, cap2,
        (ones(nt), zeros(nt)), (ones(nt), zeros(nt)),
        (2.0 .* ones(nt), zeros(nt)), (2.0 .* ones(nt), zeros(nt));
        source1=0.0,
        source2=0.0,
        bc_border1=BorderConditions(; left=Inflow(1.0), right=Outflow(), bottom=Periodic(), top=Periodic()),
        bc_border2=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Periodic(), top=Periodic()),
        scheme=Upwind1(),
    )
    res = solve_unsteady!(model, (zeros(nt), zeros(nt)), (0.0, 0.2); dt=0.02, scheme=:BE, save_history=false)
    xf = res.states[end]
    @test all(isfinite, xf)
    met = interface_flux_metrics(model, xf)
    @test met.niface > 0
    @test met.rel_abs < 1e-12
end

@testset "Masking and inactive-row identity" begin
    grid = (0.0:0.1:1.0, 0.0:0.1:1.0)
    cap = assembled_capacity(circle_moments(grid; r=0.35, cx=0.25, cy=0.25); bc=0.0)
    nt = cap.ntotal
    lay = layout_mono(nt).offsets
    model = TransportModelMono(cap, (ones(nt), ones(nt)), (zeros(nt), zeros(nt)); bc_interface=1.0)

    sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
    assemble_unsteady_mono!(sys, model, zeros(nt), 0.0, 0.1, :BE)

    @test all(isfinite, sys.A.nzval)
    @test all(isfinite, sys.b)

    LI = LinearIndices(cap.nnodes)
    checked = 0
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        halo = any(d -> I[d] == cap.nnodes[d], 1:2)
        inactiveω = halo || !(cap.buf.V[lin] > 0)
        inactiveγ = halo || !(cap.buf.Γ[lin] > 0)
        if inactiveω
            r = lay.ω[lin]
            @test sys.A[r, r] ≈ 1.0 atol=1e-12
            checked += 1
        end
        if inactiveγ
            r = lay.γ[lin]
            @test sys.A[r, r] ≈ 1.0 atol=1e-12
            checked += 1
        end
    end
    @test checked > 0
end

# ──────────────────────────────────────────────────────────────────
# Helpers for convergence / validation tests
# ──────────────────────────────────────────────────────────────────

"""Volume-weighted L1 error   ∑ V_i |u_i - u_exact_i| / ∑ V_i"""
function _l1_weighted_error(cap, u_num, u_exact)
    V = cap.buf.V
    nt = cap.ntotal
    num = 0.0
    den = 0.0
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        num += v * abs(u_num[lin] - u_exact[lin])
        den += v
    end
    return num / den
end

"""Volume-weighted total mass ∑ V_i * u_i (over active cells)"""
function _total_mass(cap, u)
    V = cap.buf.V
    nt = cap.ntotal
    s = 0.0
    LI = LinearIndices(cap.nnodes)
    N = length(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        any(d -> I[d] == cap.nnodes[d], 1:N) && continue
        v = V[lin]
        (isfinite(v) && v > 0) || continue
        s += v * u[lin]
    end
    return s
end

# ──────────────────────────────────────────────────────────────────
# Convergence order test (1D periodic sine)
# ──────────────────────────────────────────────────────────────────

@testset "Convergence order: upwind ≈ 1, centered > 1.5" begin
    dxs = [0.1, 0.05, 0.025]
    cfl  = 0.5   # dt = cfl * dx

    for (label, scheme, T_end, u_exact_fn, tscheme, expected_order) in [
        # Upwind: short time to avoid excessive dissipation; CN for 2nd-order time
        ("Upwind1", Upwind1(), 0.2,
         (x, T) -> sin(2π * (x - T)),
         :CN, 0.8),
        # Centered: full period; CN + centered spatial → O(dx²)
        ("Centered", Centered(), 1.0,
         (x, T) -> sin(2π * x),  # exact after one full period
         :CN, 1.5),
    ]
        errs = Float64[]
        for dx in dxs
            grid = (0.0:dx:1.0,)
            cap = assembled_capacity(full_moments(grid); bc=0.0)
            nt = cap.ntotal
            x = cap.xyz[1]
            u0 = sin.(2π .* x)
            u_exact = [u_exact_fn(xi, T_end) for xi in x]

            uω = (ones(nt),)
            uγ = (ones(nt),)
            bc = BorderConditions(; left=Periodic(), right=Periodic())
            dt = cfl * dx

            m = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=scheme)
            r = solve_unsteady!(m, u0, (0.0, T_end); dt=dt, scheme=tscheme, save_history=false)
            uf = omega_view(m, r.states[end])
            push!(errs, _l1_weighted_error(cap, uf, u_exact))
        end

        # Compute convergence orders between successive refinements
        orders = [log(errs[i] / errs[i+1]) / log(dxs[i] / dxs[i+1]) for i in 1:length(dxs)-1]
        avg_order = mean(orders)
        @test avg_order > expected_order
    end
end
# ──────────────────────────────────────────────────────────────────
# Mass conservation (periodic advection)
# ──────────────────────────────────────────────────────────────────

@testset "Mass conservation (periodic advection)" begin
    dx = 0.05
    grid = (0.0:dx:1.0,)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    nt = cap.ntotal
    x = cap.xyz[1]
    u0 = exp.(-100.0 .* (x .- 0.5).^2)   # Gaussian bump

    uω = (ones(nt),)
    uγ = (ones(nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())

    for scheme_tag in (Upwind1(), Centered())
        m = TransportModelMono(cap, uω, uγ; bc_border=bc, scheme=scheme_tag)
        r = solve_unsteady!(m, u0, (0.0, 0.5); dt=0.01, scheme=:BE, save_history=true)

        mass0 = _total_mass(cap, omega_view(m, r.states[1]))
        for st in r.states
            mass_t = _total_mass(cap, omega_view(m, st))
            @test abs(mass_t - mass0) / abs(mass0) < 1e-10 # machine precision for periodic BCs
        end
    end
end

# ──────────────────────────────────────────────────────────────────
# 2D rotating Gaussian — error decreases with refinement
# ──────────────────────────────────────────────────────────────────

@testset "2D rotating Gaussian error decreases" begin
    dxs = [0.1, 0.05]
    errs = Float64[]

    for dx in dxs
        grid = (0.0:dx:1.0, 0.0:dx:1.0)
        cap = assembled_capacity(full_moments(grid); bc=0.0)
        nt = cap.ntotal
        x = [cap.C_ω[i][1] for i in 1:nt]
        y = [cap.C_ω[i][2] for i in 1:nt]

        # Solid-body rotation about (0.5, 0.5) with period T=2π
        ux = -(y .- 0.5)
        uy =  (x .- 0.5)

        sigma = 0.12
        cx0, cy0 = 0.5, 0.75
        u0 = exp.(-((x .- cx0).^2 .+ (y .- cy0).^2) ./ (2 * sigma^2))
        u_exact = copy(u0)   # exact after one full revolution T=2π

        bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
        T_rev = 2π
        umax = maximum(filter(isfinite, abs.(ux))) + maximum(filter(isfinite, abs.(uy)))
        dt = 0.5 * dx / (umax + 1e-12)

        m = TransportModelMono(cap, (ux, uy), (ux, uy); bc_border=bc, scheme=Upwind1())
        r = solve_unsteady!(m, u0, (0.0, T_rev); dt=dt, scheme=:BE, save_history=false)
        uf = omega_view(m, r.states[end])
        push!(errs, _l1_weighted_error(cap, uf, u_exact))
    end

    # Error should decrease when we refine
    @test errs[2] < errs[1]
end
