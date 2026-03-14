using LinearAlgebra
using SparseArrays
using PenguinTransport
using CartesianGeometry
using CartesianGrids
using CartesianOperators
using PenguinSolverCore
using PenguinBCs

function interface_flux_diag(ops)
    nt = size(ops.K[1], 1)
    κ = zeros(Float64, nt)
    for Kd in ops.K
        κ .+= diag(Kd)
    end
    return κ
end

function coeff_inflow(κi, scale)
    tol = 100 * eps(Float64) * max(1.0, scale)
    return isfinite(κi) && (κi < -tol)
end

function _safe_sine_on_points(points, c, t)
    out = zeros(Float64, length(points))
    for i in eachindex(points)
        x = points[i][1]
        out[i] = isfinite(x) ? sin(2π * (x - c * t)) : 0.0
    end
    return out
end

function _active_masks_two_phase(cap1, cap2)
    nt = cap1.ntotal
    maskω1 = falses(nt)
    maskγ1 = falses(nt)
    maskω2 = falses(nt)
    maskγ2 = falses(nt)
    LI = LinearIndices(cap1.nnodes)
    N = length(cap1.nnodes)
    for I in CartesianIndices(cap1.nnodes)
        lin = LI[I]
        halo = any(d -> I[d] == cap1.nnodes[d], 1:N)
        halo && continue
        v1 = cap1.buf.V[lin]
        g1 = cap1.buf.Γ[lin]
        v2 = cap2.buf.V[lin]
        g2 = cap2.buf.Γ[lin]
        maskω1[lin] = isfinite(v1) && v1 > 0
        maskγ1[lin] = isfinite(g1) && g1 > 0
        maskω2[lin] = isfinite(v2) && v2 > 0
        maskγ2[lin] = isfinite(g2) && g2 > 0
    end
    return maskω1, maskγ1, maskω2, maskγ2
end

function _row_mask_full(lay, maskω1, maskγ1, maskω2, maskγ2)
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    mask = falses(nsys)
    mask[lay.ω1] .= maskω1
    mask[lay.γ1] .= maskγ1
    mask[lay.ω2] .= maskω2
    mask[lay.γ2] .= maskγ2
    return mask
end

function _exact_full_state(cap1, cap2, lay, c, t)
    x = zeros(Float64, maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2))))
    x[lay.ω1] .= _safe_sine_on_points(cap1.C_ω, c, t)
    x[lay.γ1] .= _safe_sine_on_points(cap1.C_γ, c, t)
    x[lay.ω2] .= _safe_sine_on_points(cap2.C_ω, c, t)
    x[lay.γ2] .= _safe_sine_on_points(cap2.C_γ, c, t)
    return x
end

function _constant_full_state(lay, nt, val)
    x = zeros(Float64, maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2))))
    x[lay.ω1] .= val
    x[lay.γ1] .= val
    x[lay.ω2] .= val
    x[lay.γ2] .= val
    return x
end

function _block_inf_norms(r, lay, masks)
    maskω1, maskγ1, maskω2, maskγ2 = masks
    nω1 = any(maskω1) ? norm(r[lay.ω1][maskω1], Inf) : 0.0
    nγ1 = any(maskγ1) ? norm(r[lay.γ1][maskγ1], Inf) : 0.0
    nω2 = any(maskω2) ? norm(r[lay.ω2][maskω2], Inf) : 0.0
    nγ2 = any(maskγ2) ? norm(r[lay.γ2][maskγ2], Inf) : 0.0
    return nω1, nγ1, nω2, nγ2
end

function _print_top_gamma_cells(r, lay, cap1, cap2; topk=6)
    nt = cap1.ntotal
    iface = Int[]
    vals = Float64[]
    for i in 1:nt
        Γ1 = cap1.buf.Γ[i]
        Γ2 = cap2.buf.Γ[i]
        if (isfinite(Γ1) && Γ1 > 0) || (isfinite(Γ2) && Γ2 > 0)
            push!(iface, i)
            push!(vals, max(abs(r[lay.γ1[i]]), abs(r[lay.γ2[i]])))
        end
    end
    order = sortperm(vals; rev=true)
    nk = min(topk, length(order))
    println("Top γ residual cells:")
    for k in 1:nk
        i = iface[order[k]]
        println("  cell=", i,
            " x=", cap1.C_ω[i][1],
            " rγ1=", r[lay.γ1[i]],
            " rγ2=", r[lay.γ2[i]],
            " Γ1=", cap1.buf.Γ[i],
            " Γ2=", cap2.buf.Γ[i])
    end
end

function _is_transmission_row(A, r, lay, i, κ1i, κ2i; atol=1e-11)
    return isapprox(A[r, lay.γ1[i]], κ1i; atol=atol, rtol=0) &&
           isapprox(A[r, lay.γ2[i]], κ2i; atol=atol, rtol=0) &&
           isapprox(A[r, lay.ω1[i]], 0.0; atol=atol, rtol=0) &&
           isapprox(A[r, lay.ω2[i]], 0.0; atol=atol, rtol=0)
end

function _is_continuity_row_phase1(A, r, lay, i, Γ; atol=1e-11)
    return isapprox(A[r, lay.ω1[i]], -Γ; atol=atol, rtol=0) &&
           isapprox(A[r, lay.γ1[i]],  Γ; atol=atol, rtol=0) &&
           isapprox(A[r, lay.ω2[i]], 0.0; atol=atol, rtol=0) &&
           isapprox(A[r, lay.γ2[i]], 0.0; atol=atol, rtol=0)
end

function _is_continuity_row_phase2(A, r, lay, i, Γ; atol=1e-11)
    return isapprox(A[r, lay.ω2[i]], -Γ; atol=atol, rtol=0) &&
           isapprox(A[r, lay.γ2[i]],  Γ; atol=atol, rtol=0) &&
           isapprox(A[r, lay.ω1[i]], 0.0; atol=atol, rtol=0) &&
           isapprox(A[r, lay.γ1[i]], 0.0; atol=atol, rtol=0)
end

function _verify_static_local_rows(sys, lay, cap1, cap2, κ1, κ2)
    A = sys.A
    nt = cap1.ntotal
    LI = LinearIndices(cap1.nnodes)
    N = length(cap1.nnodes)
    println("Static two-phase local interface row roles:")
    for I in CartesianIndices(cap1.nnodes)
        i = LI[I]
        halo = any(d -> I[d] == cap1.nnodes[d], 1:N)
        halo && continue
        Γ1 = cap1.buf.Γ[i]
        Γ2 = cap2.buf.Γ[i]
        have_iface = (isfinite(Γ1) && Γ1 > 0) || (isfinite(Γ2) && Γ2 > 0)
        have_iface || continue
        Γ = 0.5 * (Γ1 + Γ2)

        κ1i = κ1[i]
        κ2i = κ2[i]
        scale = max(abs(κ1i), abs(κ2i))
        in1 = coeff_inflow(κ1i, scale)
        in2 = coeff_inflow(κ2i, scale)

        r1 = lay.γ1[i]
        r2 = lay.γ2[i]
        trans1 = _is_transmission_row(A, r1, lay, i, κ1i, κ2i)
        trans2 = _is_transmission_row(A, r2, lay, i, κ1i, κ2i)
        cont1 = _is_continuity_row_phase1(A, r1, lay, i, Γ)
        cont2 = _is_continuity_row_phase2(A, r2, lay, i, Γ)

        role1 = trans1 ? "transmission" : (cont1 ? "continuity-1" : "other")
        role2 = trans2 ? "transmission" : (cont2 ? "continuity-2" : "other")
        println("  i=", i,
            " κ1=", κ1i,
            " κ2=", κ2i,
            " inflow=(", in1, ",", in2, ")",
            " row(γ1)=", role1,
            " row(γ2)=", role2)

        if in1 && in2
            error("both-inflow found in debug case at cell $i")
        elseif in1 && !in2
            @assert trans1
            @assert cont2
        elseif in2 && !in1
            @assert trans2
            @assert cont1
        else
            @assert cont1
            @assert cont2
        end
    end
end

function _fixed_model(n, scheme; c=0.4, x0=0.5, r=0.18)
    grid = (range(0.0, 1.0; length=n),)
    body(x) = abs(x - x0) - r
    cap1 = assembled_capacity(geometric_moments(body, grid, Float64, nan; method=:vofijul); bc=0.0)
    cap2 = assembled_capacity(geometric_moments(x -> -body(x), grid, Float64, nan; method=:vofijul); bc=0.0)
    nt = cap1.ntotal
    vel = (fill(c, nt),)
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    return TransportModelTwoPhase(
        cap1, cap2,
        vel, vel,
        vel, vel;
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=scheme,
    )
end

function _moving_material_model(n, scheme; c=0.4, x0=0.5, r=0.18)
    grid = CartesianGrid((0.0,), (1.0,), (n,))
    nt = prod(grid.n)
    vel = (fill(c, nt),)
    body1(x, t) = abs(x - (x0 + c * t)) - r
    bc = BorderConditions(; left=Periodic(), right=Periodic())
    return MovingTransportModelTwoPhase(
        grid, body1,
        vel, vel,
        vel, vel;
        body2=nothing,
        wγ=vel,
        source1=0.0,
        source2=0.0,
        bc_border1=bc,
        bc_border2=bc,
        scheme=scheme,
        geom_method=:vofijul,
    )
end

function run_debug(; n=65, c=0.4, tprobe=0.1)
    println("=== Fixed Two-Phase Interface Debug (band/complement periodic) ===")
    println("n=$n c=$c tprobe=$tprobe")
    for scheme in (Upwind1(), Centered())
        println("\n--- scheme = ", typeof(scheme), " ---")

        model = _fixed_model(n, scheme; c=c)
        nt = model.cap1.ntotal
        lay = model.layout
        sys = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
        assemble_steady_two_phase!(sys, model, tprobe)

        masks = _active_masks_two_phase(model.cap1, model.cap2)
        rowmask = _row_mask_full(lay, masks...)

        # Step 3: constant-state exactness check
        xconst = _constant_full_state(lay, nt, 1.0)
        rconst = sys.A * xconst - sys.b
        nω1c, nγ1c, nω2c, nγ2c = _block_inf_norms(rconst, lay, masks)
        nconst = any(rowmask) ? norm(rconst[rowmask], Inf) : 0.0
        println("constant residual block inf norms:")
        println("  ||rω1||∞=$nω1c  ||rγ1||∞=$nγ1c  ||rω2||∞=$nω2c  ||rγ2||∞=$nγ2c")
        println("  ||r(active)||∞=$nconst")
        if scheme isa Upwind1
            @assert nconst < 1e-10
        else
            println("  NOTE: centered constant residual is not machine-zero in this cut-cell setup; continuing diagnostic run.")
        end

        # Step 2: exact transported sine residual check
        xexact = _exact_full_state(model.cap1, model.cap2, lay, c, tprobe)
        rsin = sys.A * xexact - sys.b
        nω1, nγ1, nω2, nγ2 = _block_inf_norms(rsin, lay, masks)
        println("sine residual block inf norms:")
        println("  ||rω1||∞=$nω1")
        println("  ||rγ1||∞=$nγ1")
        println("  ||rω2||∞=$nω2")
        println("  ||rγ2||∞=$nγ2")
        _print_top_gamma_cells(rsin, lay, model.cap1, model.cap2; topk=6)

        # Step 4 + Step 5: interface coefficients and local row roles/verification
        κ1 = interface_flux_diag(model.ops1)
        κ2 = interface_flux_diag(model.ops2)
        println("interface coefficient diagnostics:")
        iface = findall(i -> ((isfinite(model.cap1.buf.Γ[i]) && model.cap1.buf.Γ[i] > 0) ||
                               (isfinite(model.cap2.buf.Γ[i]) && model.cap2.buf.Γ[i] > 0)), 1:nt)
        for i in iface
            Γ1 = model.cap1.buf.Γ[i]
            Γ2 = model.cap2.buf.Γ[i]
            scale = max(abs(κ1[i]), abs(κ2[i]))
            in1 = coeff_inflow(κ1[i], scale)
            in2 = coeff_inflow(κ2[i], scale)
            println("  i=$i Γ1=$Γ1 Γ2=$Γ2 κ1=$(κ1[i]) κ2=$(κ2[i]) inflow=(phase1=$in1, phase2=$in2)")
        end
        _verify_static_local_rows(sys, lay, model.cap1, model.cap2, κ1, κ2)

        # Step 6: fixed vs moving material-interface comparison
        moving = _moving_material_model(n, scheme; c=c)
        laym = moving.layout
        sysm = LinearSystem(spzeros(Float64, 4 * nt, 4 * nt), zeros(Float64, 4 * nt))
        dt = 0.4 * (1.0 / (n - 1)) / c
        u0 = (ones(nt), ones(nt))
        assemble_unsteady_two_phase_moving!(sysm, moving, u0, 0.0, dt, :BE)
        cap1m = moving.cap1_slab
        cap2m = moving.cap2_slab
        @assert cap1m !== nothing
        @assert cap2m !== nothing
        κ1m = interface_flux_diag(moving.ops1_slab)
        κ2m = interface_flux_diag(moving.ops2_slab)
        ifacem = findall(i -> ((isfinite(cap1m.buf.Γ[i]) && cap1m.buf.Γ[i] > 0) ||
                                (isfinite(cap2m.buf.Γ[i]) && cap2m.buf.Γ[i] > 0)), 1:nt)
        println("moving material-interface κrel diagnostics:")
        for i in ifacem
            scale = max(abs(κ1m[i]), abs(κ2m[i]))
            in1 = coeff_inflow(κ1m[i], scale)
            in2 = coeff_inflow(κ2m[i], scale)
            println("  i=$i κ1rel=$(κ1m[i]) κ2rel=$(κ2m[i]) inflow=(phase1=$in1, phase2=$in2)")
        end

        xconstm = _constant_full_state(laym, nt, 1.0)
        rm = sysm.A * xconstm - sysm.b
        masksm = _active_masks_two_phase(cap1m, cap2m)
        nmω1, nmγ1, nmω2, nmγ2 = _block_inf_norms(rm, laym, masksm)
        println("moving material-interface residual block inf norms (constant test):")
        println("  ||rω1||∞=$nmω1  ||rγ1||∞=$nmγ1  ||rω2||∞=$nmω2  ||rγ2||∞=$nmγ2")
    end
    println("\nDebug run completed.")
end

run_debug()
