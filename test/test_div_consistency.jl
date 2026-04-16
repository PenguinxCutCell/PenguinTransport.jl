@testset "Cut-cell divergence consistency and convergence" begin
    function _safe_radial_mode(points; cx=0.5, cy=0.5)
        out = zeros(Float64, length(points))
        for i in eachindex(points)
            x = points[i][1]
            y = length(points[i]) >= 2 ? points[i][2] : NaN
            if isfinite(x) && isfinite(y)
                r = sqrt((x - cx)^2 + (y - cy)^2)
                out[i] = sin(4π * r)
            else
                out[i] = 0.0
            end
        end
        return out
    end

    function _safe_tangent_velocity(points; cx=0.5, cy=0.5)
        ux = zeros(Float64, length(points))
        uy = zeros(Float64, length(points))
        for i in eachindex(points)
            x = points[i][1]
            y = length(points[i]) >= 2 ? points[i][2] : NaN
            if isfinite(x) && isfinite(y)
                ux[i] = -(y - cy)
                uy[i] = (x - cx)
            end
        end
        return ux, uy
    end

    @testset "Constant state preserved on active ω rows" begin
        grid = (0.0:0.05:1.0, 0.0:0.05:1.0)
        cap = assembled_capacity(circle_moments(grid); bc=0.0)
        nt = cap.ntotal
        lay = layout_mono(nt).offsets

        uω = (ones(nt), zeros(nt))
        uγ = (zeros(nt), zeros(nt))
        bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
        model = TransportModelMono(cap, uω, uγ; source=0.0, bc_border=bc, bc_interface=nothing, scheme=Centered())

        sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
        assemble_steady_mono!(sys, model, 0.0)

        xconst = ones(2 * nt)
        rω = sys.A[lay.ω, :] * xconst - sys.b[lay.ω]
        active = active_omega_mask(cap)
        scale = max(maximum(abs.(sys.A.nzval)), 1.0)
        @test norm(rω[active], Inf) <= 1e-12 * scale
    end

    @testset "Discrete divergence for φ=x on cut cells" begin
        grid = (0.0:0.05:1.0, 0.0:0.05:1.0)
        cap = assembled_capacity(circle_moments(grid); bc=0.0)
        nt = cap.ntotal
        lay = layout_mono(nt).offsets

        uω = (ones(nt), zeros(nt))
        uγ = (zeros(nt), zeros(nt))
        bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
        model = TransportModelMono(cap, uω, uγ; source=0.0, bc_border=bc, bc_interface=nothing, scheme=Centered())

        sys = LinearSystem(spzeros(Float64, 2 * nt, 2 * nt), zeros(Float64, 2 * nt))
        assemble_steady_mono!(sys, model, 0.0)

        xstate = zeros(2 * nt)
        xstate[lay.ω] .= [isfinite(cap.C_ω[i][1]) ? cap.C_ω[i][1] : 0.0 for i in 1:nt]
        xstate[lay.γ] .= [isfinite(cap.C_γ[i][1]) ? cap.C_γ[i][1] : 0.0 for i in 1:nt]

        rω = sys.A[lay.ω, :] * xstate - sys.b[lay.ω]
        active = active_omega_mask(cap)
        iface = findall(i -> active[i] && isfinite(cap.buf.Γ[i]) && cap.buf.Γ[i] > 0, 1:nt)
        @test !isempty(iface)

        target = cap.V
        rel = norm(rω[iface] .- target[iface], Inf) / max(norm(target[iface], Inf), eps(Float64))
        @test rel <= 5e-2
    end

    @testset "Centered bulk operator skew symmetry behavior" begin
        grid = (0.0:0.05:1.0, 0.0:0.05:1.0)

        cap_full = assembled_capacity(full_moments(grid); bc=0.0)
        nt_full = cap_full.ntotal
        u_full = (ones(nt_full), zeros(nt_full))
        ops_full = PenguinTransport._advection_ops_moving(cap_full, u_full, u_full; periodic=(true, true), scheme=Centered())
        C_full = reduce(+, ops_full.C)
        skew_full = C_full + C_full'
        @test norm(skew_full, Inf) <= 1e-10 * max(norm(C_full, Inf), 1.0)

        cap_if = assembled_capacity(circle_moments(grid); bc=0.0)
        nt_if = cap_if.ntotal
        u_if = (ones(nt_if), zeros(nt_if))
        ops_if = PenguinTransport._advection_ops_moving(cap_if, u_if, (zeros(nt_if), zeros(nt_if)); periodic=(true, true), scheme=Centered())
        C_if = reduce(+, ops_if.C)
        S = Matrix(C_if + C_if')
        tol = 1e-10 * max(norm(C_if, Inf), 1.0)
        active = active_omega_mask(cap_if)

        for i in 1:nt_if
            rowcol_mag = max(maximum(abs.(S[i, :])), maximum(abs.(S[:, i])))
            if rowcol_mag > tol
                @test (isfinite(cap_if.buf.Γ[i]) && cap_if.buf.Γ[i] > 0) || !active[i]
            end
        end
    end

    @testset "Manufactured convergence gate mono_fixed_interface Centered/CN" begin
        function run_case(n)
            tend = 0.1
            cx = 0.5
            cy = 0.5
            r0 = 0.22
            grid = (range(0.0, 1.0; length=n), range(0.0, 1.0; length=n))
            h = min(minimum(diff(collect(grid[1]))), minimum(diff(collect(grid[2]))))
            cap = assembled_capacity(circle_moments(grid; r=r0, cx=cx, cy=cy); bc=0.0)

            uωx, uωy = _safe_tangent_velocity(cap.C_ω; cx=cx, cy=cy)
            uγx, uγy = _safe_tangent_velocity(cap.C_γ; cx=cx, cy=cy)
            bc = BorderConditions(; left=Periodic(), right=Periodic(), bottom=Periodic(), top=Periodic())
            model = TransportModelMono(cap, (uωx, uωy), (uγx, uγy);
                source=0.0,
                bc_border=bc,
                bc_interface=nothing,
                scheme=Centered(),
            )

            u0 = _safe_radial_mode(cap.C_ω; cx=cx, cy=cy)
            umax = max(maximum(abs.(uωx)), maximum(abs.(uωy)))
            dt = 0.4 * h / max(umax, eps(Float64))
            res = solve_unsteady!(model, u0, (0.0, tend); dt=dt, scheme=:CN, save_history=false)
            uf = res.states[end][model.layout.offsets.ω]
            exact = _safe_radial_mode(cap.C_ω; cx=cx, cy=cy)
            return _l2_weighted_error(cap, uf, exact)
        end

        errs = [run_case(n) for n in (33, 65, 129)]
        p65_129 = log(errs[2] / errs[3]) / log(2)
        @test p65_129 >= 1.5
    end
end
