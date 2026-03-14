module PenguinTransport

using LinearAlgebra
using SparseArrays
using StaticArrays

using CartesianGeometry
using CartesianGeometry: GeometricMoments, geometric_moments, nan
using CartesianGrids: CartesianGrid, SpaceTimeCartesianGrid, grid1d
using CartesianOperators
using PenguinBCs
using PenguinSolverCore

export TransportModelMono
export TransportModelTwoPhase
export MovingTransportModelMono
export MovingTransportModelTwoPhase
export assemble_steady_mono!, assemble_unsteady_mono!
export assemble_steady_two_phase!, assemble_unsteady_two_phase!
export assemble_unsteady_mono_moving!, assemble_unsteady_two_phase_moving!
export solve_steady!, solve_unsteady!
export solve_unsteady_moving!
export update_advection_ops!, rebuild!
export omega1_view, gamma1_view, omega2_view, gamma2_view

"""
    TransportModelMono(cap, ops, uω, uγ; kwargs...)
    TransportModelMono(cap, uω, uγ; kwargs...)

Monophasic advection transport model on cut cells.

Unknown ordering is `(ω, γ)` where `ω` are bulk cell values and `γ` are interface values.
"""
mutable struct TransportModelMono{N,T,VTω,VTγ,ST,BCT,ICT,SCT}
	ops::AdvectionOps{N,T}
	cap::AssembledCapacity{N,T}
	uω::VTω
	uγ::VTγ
	source::ST
	bc_border::BCT
	bc_interface::ICT
	layout::UnknownLayout
	periodic::NTuple{N,Bool}
	scheme::SCT
end

layout_two_phase(nt::Int) = (
	ω1=1:nt,
	γ1=(nt + 1):(2 * nt),
	ω2=(2 * nt + 1):(3 * nt),
	γ2=(3 * nt + 1):(4 * nt),
)

"""
    TransportModelTwoPhase(cap1, cap2, ops1, ops2, u1ω, u1γ, u2ω, u2γ; kwargs...)
    TransportModelTwoPhase(cap1, cap2, u1ω, u1γ, u2ω, u2γ; kwargs...)

Two-phase advection transport model with phase-wise cut-cell capacities and flux-coupled interface rows.

Unknown ordering is `(ω1, γ1, ω2, γ2)`.
"""
mutable struct TransportModelTwoPhase{
	N,T,
	OPS1,OPS2,
	VT1ω,VT1γ,VT2ω,VT2γ,
	ST1,ST2,
	BC1,BC2,
	LT,SCT
}
	ops1::OPS1
	ops2::OPS2
	cap1::AssembledCapacity{N,T}
	cap2::AssembledCapacity{N,T}
	u1ω::VT1ω
	u1γ::VT1γ
	u2ω::VT2ω
	u2γ::VT2γ
	source1::ST1
	source2::ST2
	bc_border1::BC1
	bc_border2::BC2
	layout::LT
	periodic1::NTuple{N,Bool}
	periodic2::NTuple{N,Bool}
	scheme::SCT
end

"""
    MovingTransportModelMono(grid, body, uω, uγ; kwargs...)

Moving-geometry monophasic advection transport model assembled from space-time slabs.

Unknown ordering is `(ω, γ)` and interface inflow/outflow decisions use
relative interface speed `(uγ - wγ)·nγ`.
"""
mutable struct MovingTransportModelMono{N,T,VTω,VTγ,WTγ,ST,BCT,ICT,LT,SCT,BT}
	grid::CartesianGrid{N,T}
	body::BT
	uω::VTω
	uγ::VTγ
	wγ::WTγ
	source::ST
	bc_border::BCT
	bc_interface::ICT
	layout::LT
	periodic::NTuple{N,Bool}
	scheme::SCT
	geom_method::Symbol
	cap_slab::Union{Nothing,AssembledCapacity{N,T}}
	ops_slab::Union{Nothing,AdvectionOps{N,T}}
	Vn::Vector{T}
	Vn1::Vector{T}
end

"""
    MovingTransportModelTwoPhase(grid, body1, u1ω, u1γ, u2ω, u2γ; kwargs...)

Moving-geometry two-phase advection transport model assembled from space-time slabs.

Unknown ordering is `(ω1, γ1, ω2, γ2)` and local interface inflow/outflow
decisions use relative interface speeds `(u1γ - wγ)·n1γ` and `(u2γ - wγ)·n2γ`.
"""
mutable struct MovingTransportModelTwoPhase{
	N,T,
	B1T,B2T,
	VT1ω,VT1γ,VT2ω,VT2γ,WTγ,
	ST1,ST2,
	BC1,BC2,
	LT,SCT
}
	grid::CartesianGrid{N,T}
	body1::B1T
	body2::B2T
	u1ω::VT1ω
	u1γ::VT1γ
	u2ω::VT2ω
	u2γ::VT2γ
	wγ::WTγ
	source1::ST1
	source2::ST2
	bc_border1::BC1
	bc_border2::BC2
	layout::LT
	periodic1::NTuple{N,Bool}
	periodic2::NTuple{N,Bool}
	scheme::SCT
	geom_method::Symbol
	cap1_slab::Union{Nothing,AssembledCapacity{N,T}}
	ops1_slab::Union{Nothing,AdvectionOps{N,T}}
	cap2_slab::Union{Nothing,AssembledCapacity{N,T}}
	ops2_slab::Union{Nothing,AdvectionOps{N,T}}
	V1n::Vector{T}
	V1n1::Vector{T}
	V2n::Vector{T}
	V2n1::Vector{T}
end

function TransportModelMono(
	cap::AssembledCapacity{N,T},
	ops::AdvectionOps{N,T},
	uω,
	uγ;
	source=((args...) -> zero(T)),
	bc_border::BorderConditions=BorderConditions(),
	bc_interface=nothing,
	layout::UnknownLayout=layout_mono(cap.ntotal),
	periodic::NTuple{N,Bool}=periodic_flags(bc_border, N),
	scheme::AdvectionScheme=Centered(),
) where {N,T}
	return TransportModelMono{N,T,typeof(uω),typeof(uγ),typeof(source),typeof(bc_border),typeof(bc_interface),typeof(scheme)}(
		ops,
		cap,
		uω,
		uγ,
		source,
		bc_border,
		bc_interface,
		layout,
		periodic,
		scheme,
	)
end

function _validate_two_phase_caps(cap1::AssembledCapacity{N,T}, cap2::AssembledCapacity{N,T}) where {N,T}
	cap1.ntotal == cap2.ntotal || throw(ArgumentError("cap1.ntotal ($(cap1.ntotal)) must match cap2.ntotal ($(cap2.ntotal))"))
	cap1.nnodes == cap2.nnodes || throw(ArgumentError("cap1.nnodes ($(cap1.nnodes)) must match cap2.nnodes ($(cap2.nnodes))"))
	return nothing
end

function TransportModelTwoPhase(
	cap1::AssembledCapacity{N,T},
	cap2::AssembledCapacity{N,T},
	ops1::AdvectionOps{N,T},
	ops2::AdvectionOps{N,T},
	u1ω,
	u1γ,
	u2ω,
	u2γ;
	source1=((args...) -> zero(T)),
	source2=((args...) -> zero(T)),
	bc_border1::BorderConditions=BorderConditions(),
	bc_border2::BorderConditions=BorderConditions(),
	layout=layout_two_phase(cap1.ntotal),
	periodic1::NTuple{N,Bool}=periodic_flags(bc_border1, N),
	periodic2::NTuple{N,Bool}=periodic_flags(bc_border2, N),
	scheme::AdvectionScheme=Centered(),
) where {N,T}
	_validate_two_phase_caps(cap1, cap2)
	return TransportModelTwoPhase{
		N,T,
		typeof(ops1),typeof(ops2),
		typeof(u1ω),typeof(u1γ),typeof(u2ω),typeof(u2γ),
		typeof(source1),typeof(source2),
		typeof(bc_border1),typeof(bc_border2),
		typeof(layout),typeof(scheme),
	}(
		ops1,
		ops2,
		cap1,
		cap2,
		u1ω,
		u1γ,
		u2ω,
		u2γ,
		source1,
		source2,
		bc_border1,
		bc_border2,
		layout,
		periodic1,
		periodic2,
		scheme,
	)
end

function TransportModelTwoPhase(
	cap1::AssembledCapacity{N,T},
	cap2::AssembledCapacity{N,T},
	u1ω,
	u1γ,
	u2ω,
	u2γ;
	source1=((args...) -> zero(T)),
	source2=((args...) -> zero(T)),
	bc_border1::BorderConditions=BorderConditions(),
	bc_border2::BorderConditions=BorderConditions(),
	layout=layout_two_phase(cap1.ntotal),
	periodic1::NTuple{N,Bool}=periodic_flags(bc_border1, N),
	periodic2::NTuple{N,Bool}=periodic_flags(bc_border2, N),
	scheme::AdvectionScheme=Centered(),
) where {N,T}
	_validate_two_phase_caps(cap1, cap2)
	u1ωv, u1γv = _velocity_values(cap1, u1ω, u1γ, zero(T))
	u2ωv, u2γv = _velocity_values(cap2, u2ω, u2γ, zero(T))
	ops1 = _advection_ops_moving(cap1, u1ωv, u1γv; periodic=periodic1, scheme=scheme)
	ops2 = _advection_ops_moving(cap2, u2ωv, u2γv; periodic=periodic2, scheme=scheme)
	return TransportModelTwoPhase(
		cap1,
		cap2,
		ops1,
		ops2,
		u1ω,
		u1γ,
		u2ω,
		u2γ;
		source1=source1,
		source2=source2,
		bc_border1=bc_border1,
		bc_border2=bc_border2,
		layout=layout,
		periodic1=periodic1,
		periodic2=periodic2,
		scheme=scheme,
	)
end

function TransportModelMono(
	cap::AssembledCapacity{N,T},
	uω,
	uγ;
	source=((args...) -> zero(T)),
	bc_border::BorderConditions=BorderConditions(),
	bc_interface=nothing,
	layout::UnknownLayout=layout_mono(cap.ntotal),
	periodic::NTuple{N,Bool}=periodic_flags(bc_border, N),
	scheme::AdvectionScheme=Centered(),
) where {N,T}
	uωv, uγv = _velocity_values(cap, uω, uγ, zero(T))
	ops = _advection_ops_moving(cap, uωv, uγv; periodic=periodic, scheme=scheme)
	return TransportModelMono(
		cap,
		ops,
		uω,
		uγ;
		source=source,
		bc_border=bc_border,
		bc_interface=bc_interface,
		layout=layout,
		periodic=periodic,
		scheme=scheme,
	)
end

function _eval_fun_or_const(v, x::SVector{N,T}, t::T) where {N,T}
	if v isa Number
		return convert(T, v)
	elseif v isa Ref
		return _eval_fun_or_const(v[], x, t)
	elseif v isa Function
		if applicable(v, x..., t)
			return convert(T, v(x..., t))
		elseif applicable(v, x...)
			return convert(T, v(x...))
		end
	end
	throw(ArgumentError("callback/value must be numeric, Ref, (x...), or (x..., t)"))
end

"""
    MovingTransportModelMono(grid, body, uω, uγ; source, bc_border, bc_interface, layout, periodic, scheme, wγ, geom_method)

Build a moving monophasic transport model.

- `body` is a level-set callback accepting `(x...)` or `(x..., t)`.
- `wγ` is the interface geometry velocity sampled on `C_γ`.
- Per-step slab caches (`cap_slab`, `ops_slab`, `Vn`, `Vn1`) are updated by moving assembly.
"""
function MovingTransportModelMono(
	grid::CartesianGrid{N,T},
	body,
	uω,
	uγ;
	wγ=ntuple(_ -> zero(T), N),
	source=((args...) -> zero(T)),
	bc_border::BorderConditions=BorderConditions(),
	bc_interface=nothing,
	layout::UnknownLayout=layout_mono(prod(grid.n)),
	periodic::NTuple{N,Bool}=periodic_flags(bc_border, N),
	scheme::AdvectionScheme=Centered(),
	geom_method::Symbol=:vofijul,
) where {N,T}
	nt = prod(grid.n)
	return MovingTransportModelMono{
		N,T,
		typeof(uω),typeof(uγ),typeof(wγ),typeof(source),
		typeof(bc_border),typeof(bc_interface),typeof(layout),typeof(scheme),typeof(body),
	}(
		grid,
		body,
		uω,
		uγ,
		wγ,
		source,
		bc_border,
		bc_interface,
		layout,
		periodic,
		scheme,
		geom_method,
		nothing,
		nothing,
		zeros(T, nt),
		zeros(T, nt),
	)
end

"""
    MovingTransportModelTwoPhase(grid, body1, u1ω, u1γ, u2ω, u2γ; source1, source2, body2, bc_border1, bc_border2, layout, periodic1, periodic2, scheme, wγ, geom_method)

Build a moving two-phase transport model.

- If `body2` is omitted, phase 2 uses `-body1`.
- `wγ` is the interface geometry velocity sampled on `C_γ`.
- Per-step slab caches (`cap*_slab`, `ops*_slab`, `V*n`, `V*n1`) are updated by moving assembly.
"""
function MovingTransportModelTwoPhase(
	grid::CartesianGrid{N,T},
	body1,
	u1ω,
	u1γ,
	u2ω,
	u2γ;
	wγ=ntuple(_ -> zero(T), N),
	source1=((args...) -> zero(T)),
	source2=((args...) -> zero(T)),
	body2=nothing,
	bc_border1::BorderConditions=BorderConditions(),
	bc_border2::BorderConditions=BorderConditions(),
	layout=layout_two_phase(prod(grid.n)),
	periodic1::NTuple{N,Bool}=periodic_flags(bc_border1, N),
	periodic2::NTuple{N,Bool}=periodic_flags(bc_border2, N),
	scheme::AdvectionScheme=Centered(),
	geom_method::Symbol=:vofijul,
) where {N,T}
	nt = prod(grid.n)
	return MovingTransportModelTwoPhase{
		N,T,
		typeof(body1),typeof(body2),
		typeof(u1ω),typeof(u1γ),typeof(u2ω),typeof(u2γ),typeof(wγ),
		typeof(source1),typeof(source2),
		typeof(bc_border1),typeof(bc_border2),
		typeof(layout),typeof(scheme),
	}(
		grid,
		body1,
		body2,
		u1ω,
		u1γ,
		u2ω,
		u2γ,
		wγ,
		source1,
		source2,
		bc_border1,
		bc_border2,
		layout,
		periodic1,
		periodic2,
		scheme,
		geom_method,
		nothing,
		nothing,
		nothing,
		nothing,
		zeros(T, nt),
		zeros(T, nt),
		zeros(T, nt),
		zeros(T, nt),
	)
end

function _source_values(cap::AssembledCapacity{N,T}, source, t::T) where {N,T}
	out = Vector{T}(undef, cap.ntotal)
	@inbounds for i in eachindex(out)
		out[i] = _eval_fun_or_const(source, cap.C_ω[i], t)
	end
	return out
end

function _component_values(cap::AssembledCapacity{N,T}, c, xpts::Vector{SVector{N,T}}, t::T) where {N,T}
	nt = cap.ntotal
	if c isa AbstractVector
		length(c) == nt || throw(ArgumentError("velocity vector length must be $nt"))
		return Vector{T}(c)
	elseif c isa Number || c isa Ref || c isa Function
		out = Vector{T}(undef, nt)
		@inbounds for i in 1:nt
			out[i] = _eval_fun_or_const(c, xpts[i], t)
		end
		return out
	end
	throw(ArgumentError("unsupported velocity component type $(typeof(c))"))
end

function _as_velocity_components(u, N::Int)
	if u isa Tuple
		length(u) == N || throw(ArgumentError("velocity tuple must contain $N components"))
		return u
	elseif u isa AbstractVector
		length(u) == N || throw(ArgumentError("velocity container must contain $N components"))
		return Tuple(u)
	end
	throw(ArgumentError("velocity field must be a tuple/vector of $N components"))
end

function _velocity_tuple_values(cap::AssembledCapacity{N,T}, u, xpts::Vector{SVector{N,T}}, t::T) where {N,T}
	comps = _as_velocity_components(u, N)
	return ntuple(d -> _component_values(cap, comps[d], xpts, t), N)
end

function _velocity_values(cap::AssembledCapacity{N,T}, uω, uγ, t::T) where {N,T}
	return _velocity_tuple_values(cap, uω, cap.C_ω, t), _velocity_tuple_values(cap, uγ, cap.C_γ, t)
end

function _relative_interface_velocity(
	uγ::NTuple{N,Vector{T}},
	wγ::NTuple{N,Vector{T}},
) where {N,T}
	return ntuple(d -> uγ[d] .- wγ[d], N)
end

function _shift_inactive_couplings_to_diagonal!(
	C::SparseMatrixCSC{T,Int},
	activeω::BitVector,
) where {T}
	n = size(C, 1)
	size(C, 2) == n || throw(ArgumentError("operator block must be square"))
	length(activeω) == n || throw(ArgumentError("active mask length must match operator size"))

	add_diag = zeros(T, n)
	@inbounds for j in 1:n
		col_active = activeω[j]
		for p in nzrange(C, j)
			i = C.rowval[p]
			if activeω[i] && !col_active
				v = C.nzval[p]
				if v != zero(T)
					add_diag[i] += v
					C.nzval[p] = zero(T)
				end
			end
		end
	end
	dropzeros!(C)

	@inbounds for i in 1:n
		v = add_diag[i]
		v == zero(T) && continue
		C[i, i] = C[i, i] + v
	end
	return C
end

function _advection_ops_moving(
	cap::AssembledCapacity{N,T},
	uω::NTuple{N,<:AbstractVector{T}},
	uγ::NTuple{N,<:AbstractVector{T}};
	periodic::NTuple{N,Bool},
	scheme::AdvectionScheme,
) where {N,T}
	nt = cap.ntotal
	uγflat = Vector{T}(undef, N * nt)
	@inbounds for d in 1:N
		copyto!(uγflat, (d - 1) * nt + 1, uγ[d], 1, nt)
	end
	G, H, _, nnodes, D_m, D_p, _, S_p = CartesianOperators.build_GHW(cap; periodic=periodic)
	activeω, _ = _cell_activity_masks(cap)
	C = ntuple(d -> begin
		a = cap.buf.A[d] .* uω[d]
		if scheme isa Upwind1
			a⁺ = max.(a, zero(T))
			a⁻ = min.(a, zero(T))
			return spdiagm(0 => a⁺) * D_m[d] + spdiagm(0 => a⁻) * D_p[d]
		elseif scheme isa Centered
			Cd = convert(T, 0.5) * (spdiagm(0 => a) * D_m[d] + spdiagm(0 => a) * D_p[d])
			return _shift_inactive_couplings_to_diagonal!(Cd, activeω)
		end
		throw(ArgumentError("unknown advection scheme $(typeof(scheme))"))
	end, N)
	Htuγ = H' * uγflat
	K = ntuple(d -> spdiagm(0 => S_p[d] * Htuγ), N)
	return AdvectionOps{N,T}(C, K, H, cap.V, nnodes)
end

function _interface_flux_diag(ops::AdvectionOps{N,T}) where {N,T}
	nt = size(ops.K[1], 1)
	κ = zeros(T, nt)
	@inbounds for d in 1:N
		κ .+= diag(ops.K[d])
	end
	return κ
end

function _split_interface_flux(κ::AbstractVector{T}, ::Centered) where {T}
	half = convert(T, 0.5)
	return half .* κ, half .* κ
end

function _split_interface_flux(κ::AbstractVector{T}, ::Upwind1) where {T}
	return max.(κ, zero(T)), min.(κ, zero(T))
end

function _split_interface_flux(::AbstractVector, scheme::AdvectionScheme)
	throw(ArgumentError("unknown advection scheme $(typeof(scheme))"))
end

_relative_speed_atol(::Type{T}) where {T} = convert(T, 100) * eps(T)

function _eval_levelset_time(body, x::SVector{N,T}, t::T) where {N,T}
	if applicable(body, x..., t)
		return convert(T, body(x..., t))
	elseif applicable(body, x...)
		return convert(T, body(x...))
	end
	throw(ArgumentError("level-set callback must accept (x...) or (x..., t)"))
end

function _space_moments_at_time(
	model::MovingTransportModelMono{N,T},
	xyz_space::NTuple{N,AbstractVector{T}},
	t::T,
) where {N,T}
	body_t = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
	return geometric_moments(body_t, xyz_space, T, nan; method=model.geom_method)
end

function _phase_levelset_value(model::MovingTransportModelTwoPhase{N,T}, phase::Int, x::SVector{N,T}, t::T) where {N,T}
	if phase == 1
		return _eval_levelset_time(model.body1, x, t)
	elseif phase == 2
		if model.body2 === nothing
			return -_eval_levelset_time(model.body1, x, t)
		end
		return _eval_levelset_time(model.body2, x, t)
	end
	throw(ArgumentError("phase must be 1 or 2"))
end

function _space_moments_at_time(
	model::MovingTransportModelTwoPhase{N,T},
	xyz_space::NTuple{N,AbstractVector{T}},
	t::T,
	phase::Int,
) where {N,T}
	body_t = (x...) -> _phase_levelset_value(model, phase, SVector{N,T}(x), t)
	return geometric_moments(body_t, xyz_space, T, nan; method=model.geom_method)
end

function _slice_spacetime_to_space(
	vec_st::AbstractVector,
	nn_space::NTuple{N,Int},
	nt::Int,
	it::Int,
) where {N}
	dims_st = (nn_space..., nt)
	li_st = LinearIndices(dims_st)
	li_sp = LinearIndices(nn_space)
	out = similar(vec_st, prod(nn_space))
	@inbounds for I in CartesianIndices(nn_space)
		out[li_sp[I]] = vec_st[li_st[Tuple(I)..., it]]
	end
	return out
end

function _reduce_slab_to_space(
	m_st::GeometricMoments{N1,T},
	nn_space::NTuple{N,Int},
) where {N1,N,T}
	N1 == N + 1 || throw(ArgumentError("expected slab moments dimension $(N + 1), got $N1"))
	nt = length(m_st.xyz[N1])
	nt == 2 || throw(ArgumentError("space-time reduction expects 2 time nodes, got $nt"))

	V = _slice_spacetime_to_space(m_st.V, nn_space, nt, 1)
	Γ = _slice_spacetime_to_space(m_st.interface_measure, nn_space, nt, 1)
	ctype = _slice_spacetime_to_space(m_st.cell_type, nn_space, nt, 1)
	A = ntuple(d -> _slice_spacetime_to_space(m_st.A[d], nn_space, nt, 1), N)
	B = ntuple(d -> _slice_spacetime_to_space(m_st.B[d], nn_space, nt, 1), N)
	W = ntuple(d -> _slice_spacetime_to_space(m_st.W[d], nn_space, nt, 1), N)

	bary_st = _slice_spacetime_to_space(m_st.barycenter, nn_space, nt, 1)
	baryγ_st = _slice_spacetime_to_space(m_st.barycenter_interface, nn_space, nt, 1)
	nγ_st = _slice_spacetime_to_space(m_st.interface_normal, nn_space, nt, 1)

	bary = Vector{SVector{N,T}}(undef, length(V))
	baryγ = Vector{SVector{N,T}}(undef, length(V))
	nγ = Vector{SVector{N,T}}(undef, length(V))
	@inbounds for i in eachindex(V)
		bi = bary_st[i]
		bγi = baryγ_st[i]
		ni = nγ_st[i]
		bary[i] = SVector{N,T}(ntuple(d -> bi[d], N))
		baryγ[i] = SVector{N,T}(ntuple(d -> bγi[d], N))
		nγ[i] = SVector{N,T}(ntuple(d -> ni[d], N))
	end

	xyz = ntuple(d -> collect(T, m_st.xyz[d]), N)
	return GeometricMoments(V, bary, Γ, ctype, baryγ, nγ, A, B, W, xyz)
end

function _build_moving_slab!(
	model::MovingTransportModelMono{N,T},
	t::T,
	dt::T,
) where {N,T}
	xyz_space = grid1d(model.grid)
	moms_n = _space_moments_at_time(model, xyz_space, t)
	moms_n1 = _space_moments_at_time(model, xyz_space, t + dt)

	stgrid = SpaceTimeCartesianGrid(model.grid, T[t, t + dt])
	xyz_st = grid1d(stgrid)
	body_st = (x...) -> begin
		xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
		_eval_levelset_time(model.body, xs, convert(T, x[N + 1]))
	end
	moms_st = geometric_moments(body_st, xyz_st, T, nan; method=model.geom_method)
	moms_slab = _reduce_slab_to_space(moms_st, model.grid.n)
	cap_slab = assembled_capacity(moms_slab; bc=zero(T))
	uωv, uγv = _velocity_values(cap_slab, model.uω, model.uγ, t)
	wγv = _velocity_tuple_values(cap_slab, model.wγ, cap_slab.C_γ, t)
	uγrel = _relative_interface_velocity(uγv, wγv)
	ops_slab = _advection_ops_moving(cap_slab, uωv, uγrel; periodic=model.periodic, scheme=model.scheme)

	model.cap_slab = cap_slab
	model.ops_slab = ops_slab
	model.Vn .= moms_n.V
	model.Vn1 .= moms_n1.V
	return model
end

function _build_moving_slab!(
	model::MovingTransportModelTwoPhase{N,T},
	t::T,
	dt::T,
) where {N,T}
	xyz_space = grid1d(model.grid)
	moms1_n = _space_moments_at_time(model, xyz_space, t, 1)
	moms1_n1 = _space_moments_at_time(model, xyz_space, t + dt, 1)
	moms2_n = _space_moments_at_time(model, xyz_space, t, 2)
	moms2_n1 = _space_moments_at_time(model, xyz_space, t + dt, 2)

	stgrid = SpaceTimeCartesianGrid(model.grid, T[t, t + dt])
	xyz_st = grid1d(stgrid)
	body1_st = (x...) -> begin
		xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
		_phase_levelset_value(model, 1, xs, convert(T, x[N + 1]))
	end
	body2_st = (x...) -> begin
		xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
		_phase_levelset_value(model, 2, xs, convert(T, x[N + 1]))
	end
	moms1_st = geometric_moments(body1_st, xyz_st, T, nan; method=model.geom_method)
	moms2_st = geometric_moments(body2_st, xyz_st, T, nan; method=model.geom_method)
	moms1_slab = _reduce_slab_to_space(moms1_st, model.grid.n)
	moms2_slab = _reduce_slab_to_space(moms2_st, model.grid.n)

	cap1_slab = assembled_capacity(moms1_slab; bc=zero(T))
	cap2_slab = assembled_capacity(moms2_slab; bc=zero(T))
	u1ωv, u1γv = _velocity_values(cap1_slab, model.u1ω, model.u1γ, t)
	u2ωv, u2γv = _velocity_values(cap2_slab, model.u2ω, model.u2γ, t)
	wγv = _velocity_tuple_values(cap1_slab, model.wγ, cap1_slab.C_γ, t)
	u1γrel = _relative_interface_velocity(u1γv, wγv)
	u2γrel = _relative_interface_velocity(u2γv, wγv)
	ops1_slab = _advection_ops_moving(cap1_slab, u1ωv, u1γrel; periodic=model.periodic1, scheme=model.scheme)
	ops2_slab = _advection_ops_moving(cap2_slab, u2ωv, u2γrel; periodic=model.periodic2, scheme=model.scheme)

	model.cap1_slab = cap1_slab
	model.ops1_slab = ops1_slab
	model.cap2_slab = cap2_slab
	model.ops2_slab = ops2_slab
	model.V1n .= moms1_n.V
	model.V1n1 .= moms1_n1.V
	model.V2n .= moms2_n.V
	model.V2n1 .= moms2_n1.V
	return model
end

function _value_time_dependent(v, x::SVector{N,T}) where {N,T}
	if v isa Ref
		return _value_time_dependent(v[], x)
	end
	return v isa Function && applicable(v, x..., zero(T))
end

function _vel_time_dependent(u, x::SVector{N,T}) where {N,T}
	comps = _as_velocity_components(u, N)
	for c in comps
		_value_time_dependent(c, x) && return true
	end
	return false
end

function _adv_border_time_dependent(bc::BorderConditions, x::SVector{N,T}) where {N,T}
	for side_bc in values(bc.borders)
		if side_bc isa Inflow
			_value_time_dependent(side_bc.value, x) && return true
		end
	end
	return false
end

_interface_time_dependent(::Nothing, x::SVector{N,T}) where {N,T} = false
_interface_time_dependent(bc_interface::Inflow, x::SVector{N,T}) where {N,T} = _value_time_dependent(bc_interface.value, x)
_interface_time_dependent(::Outflow, x::SVector{N,T}) where {N,T} = false
_interface_time_dependent(::Periodic, x::SVector{N,T}) where {N,T} = false
_interface_time_dependent(bc_interface::Dirichlet, x::SVector{N,T}) where {N,T} = _value_time_dependent(bc_interface.value, x)
_interface_time_dependent(bc_interface, x::SVector{N,T}) where {N,T} = _value_time_dependent(bc_interface, x)

function _interface_inflow_value(bc_interface, x::SVector{N,T}, t::T) where {N,T}
	if bc_interface === nothing
		return nothing
	elseif bc_interface isa Inflow
		return _eval_fun_or_const(bc_interface.value, x, t)
	elseif bc_interface isa Dirichlet
		return _eval_fun_or_const(bc_interface.value, x, t)
	elseif bc_interface isa Outflow || bc_interface isa Periodic
		return nothing
	elseif bc_interface isa AbstractBoundary
		throw(ArgumentError("unsupported bc_interface type $(typeof(bc_interface)); expected nothing, Inflow(value), Dirichlet(value), or scalar/callback"))
	end
	return _eval_fun_or_const(bc_interface, x, t)
end

function _insert_block!(A::SparseMatrixCSC{T,Int}, rows::UnitRange{Int}, cols::UnitRange{Int}, B::SparseMatrixCSC{T,Int}) where {T}
	size(B, 1) == length(rows) || throw(DimensionMismatch("block rows do not match target range"))
	size(B, 2) == length(cols) || throw(DimensionMismatch("block cols do not match target range"))
	@inbounds for j in 1:size(B, 2)
		for p in nzrange(B, j)
			i = B.rowval[p]
			A[rows[i], cols[j]] = A[rows[i], cols[j]] + B.nzval[p]
		end
	end
	return A
end

function _insert_vec!(b::Vector{T}, rows::UnitRange{Int}, v::Vector{T}) where {T}
	length(v) == length(rows) || throw(DimensionMismatch("vector block length mismatch"))
	@inbounds for i in eachindex(v)
		b[rows[i]] += v[i]
	end
	return b
end

function _set_row_identity!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, row::Int, value::T=zero(T)) where {T}
	@inbounds for j in 1:size(A, 2)
		A[row, j] = zero(T)
	end
	A[row, row] = one(T)
	b[row] = value
	return A, b
end

function _scale_rows!(A::SparseMatrixCSC{T,Int}, rows::UnitRange{Int}, α::T) where {T}
	α == one(T) && return A
	r1 = first(rows)
	r2 = last(rows)
	@inbounds for j in 1:size(A, 2)
		for p in nzrange(A, j)
			i = A.rowval[p]
			if r1 <= i <= r2
				A.nzval[p] *= α
			end
		end
	end
	return A
end

function _apply_row_identity_constraints!(
	A::SparseMatrixCSC{T,Int},
	b::Vector{T},
	active_rows::BitVector,
) where {T}
	n = size(A, 1)
	size(A, 2) == n || throw(ArgumentError("row-identity constraints require square matrix"))
	length(b) == n || throw(ArgumentError("rhs length mismatch"))
	length(active_rows) == n || throw(ArgumentError("active row mask length mismatch"))

	p = Vector{T}(undef, n)
	@inbounds for i in 1:n
		ai = active_rows[i]
		p[i] = ai ? zero(T) : one(T)
		ai || (b[i] = zero(T))
	end

	@inbounds for j in 1:size(A, 2)
		aj = active_rows[j]
		for k in nzrange(A, j)
			if !(aj && active_rows[A.rowval[k]])
				A.nzval[k] = zero(T)
			end
		end
	end
	dropzeros!(A)
	Aout = A + spdiagm(0 => p)
	return Aout, b
end

function _cell_activity_masks(cap::AssembledCapacity{N,T}) where {N,T}
	nt = cap.ntotal
	activeω = BitVector(undef, nt)
	activeγ = BitVector(undef, nt)
	LI = LinearIndices(cap.nnodes)
	for I in CartesianIndices(cap.nnodes)
		lin = LI[I]
		halo = any(d -> I[d] == cap.nnodes[d], 1:N)
		if halo
			activeω[lin] = false
			activeγ[lin] = false
			continue
		end
		v = cap.buf.V[lin]
		γ = cap.buf.Γ[lin]
		activeω[lin] = isfinite(v) && v > zero(T)
		activeγ[lin] = isfinite(γ) && γ > zero(T)
	end
	return activeω, activeγ
end

function _write_row_activity!(active::BitVector, rows::UnitRange{Int}, mask::BitVector)
	length(rows) == length(mask) || throw(ArgumentError("row range length ($(length(rows))) must match mask length ($(length(mask)))"))
	@inbounds for i in eachindex(mask)
		active[rows[i]] = mask[i]
	end
	return active
end

function _mono_row_activity(
	cap::AssembledCapacity{N,T},
	lay,
) where {N,T}
	activeω, activeγ = _cell_activity_masks(cap)
	nsys = maximum((last(lay.ω), last(lay.γ)))
	active = falses(nsys)
	_write_row_activity!(active, lay.ω, activeω)
	_write_row_activity!(active, lay.γ, activeγ)
	return active
end

function _two_phase_row_activity(
	cap1::AssembledCapacity{N,T},
	cap2::AssembledCapacity{N,T},
	lay,
) where {N,T}
	active1ω, active1γ = _cell_activity_masks(cap1)
	active2ω, active2γ = _cell_activity_masks(cap2)
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
	active = falses(nsys)
	_write_row_activity!(active, lay.ω1, active1ω)
	_write_row_activity!(active, lay.γ1, active1γ)
	_write_row_activity!(active, lay.ω2, active2ω)
	_write_row_activity!(active, lay.γ2, active2γ)
	return active
end

function _validate_mono_layout(cap::AssembledCapacity, lay)
	nt = cap.ntotal
	length(lay.ω) == nt || throw(ArgumentError("layout ω length ($(length(lay.ω))) must match cap.ntotal ($nt)"))
	length(lay.γ) == nt || throw(ArgumentError("layout γ length ($(length(lay.γ))) must match cap.ntotal ($nt)"))
	return nothing
end

function _validate_two_phase_layout(cap1::AssembledCapacity, cap2::AssembledCapacity, lay)
	nt1 = cap1.ntotal
	nt2 = cap2.ntotal
	nt1 == nt2 || throw(ArgumentError("phase capacities must have identical ntotal; got $nt1 and $nt2"))
	length(lay.ω1) == nt1 || throw(ArgumentError("layout ω1 length ($(length(lay.ω1))) must match cap1.ntotal ($nt1)"))
	length(lay.γ1) == nt1 || throw(ArgumentError("layout γ1 length ($(length(lay.γ1))) must match cap1.ntotal ($nt1)"))
	length(lay.ω2) == nt2 || throw(ArgumentError("layout ω2 length ($(length(lay.ω2))) must match cap2.ntotal ($nt2)"))
	length(lay.γ2) == nt2 || throw(ArgumentError("layout γ2 length ($(length(lay.γ2))) must match cap2.ntotal ($nt2)"))
	return nothing
end

function _interface_closure(
	model::TransportModelMono{N,T},
	κ::AbstractVector{T},
	t::T,
) where {N,T}
	nt = model.cap.ntotal
	length(κ) == nt || throw(DimensionMismatch("interface coefficient length must be $nt"))
	a21 = zeros(T, nt)
	a22 = zeros(T, nt)
	b2 = zeros(T, nt)
	LI = LinearIndices(model.cap.nnodes)

	for I in CartesianIndices(model.cap.nnodes)
		lin = LI[I]
		any(d -> I[d] == model.cap.nnodes[d], 1:N) && continue
		Γ = model.cap.buf.Γ[lin]
		if !(isfinite(Γ) && Γ > zero(T))
			continue
		end

		κi = κ[lin]
		if _is_inflow_coeff(κi, abs(κi))
			g = _interface_inflow_value(model.bc_interface, model.cap.C_γ[lin], t)
			if g !== nothing
				a22[lin] = Γ
				b2[lin] = Γ * g
				continue
			end
		end
		a21[lin] = -Γ
		a22[lin] = Γ
	end

	return spdiagm(0 => a21), spdiagm(0 => a22), b2
end

function _is_inflow_coeff(κi::T, scale::T) where {T}
	isfinite(κi) || return false
	tol = _relative_speed_atol(T) * max(one(T), scale)
	return κi < -tol
end

function _interface_closure_moving(
	model::MovingTransportModelMono{N,T},
	cap::AssembledCapacity{N,T},
	κrel::AbstractVector{T},
	t::T,
) where {N,T}
	nt = cap.ntotal
	length(κrel) == nt || throw(DimensionMismatch("relative interface coefficient length must be $nt"))
	a21 = zeros(T, nt)
	a22 = zeros(T, nt)
	b2 = zeros(T, nt)
	LI = LinearIndices(cap.nnodes)

	for I in CartesianIndices(cap.nnodes)
		lin = LI[I]
		any(d -> I[d] == cap.nnodes[d], 1:N) && continue
		Γ = cap.buf.Γ[lin]
		if !(isfinite(Γ) && Γ > zero(T))
			continue
		end

		κi = κrel[lin]
		if _is_inflow_coeff(κi, abs(κi))
			g = _interface_inflow_value(model.bc_interface, cap.C_γ[lin], t)
			if g !== nothing
				a22[lin] = Γ
				b2[lin] = Γ * g
				continue
			end
		end
		a21[lin] = -Γ
		a22[lin] = Γ
	end

	return spdiagm(0 => a21), spdiagm(0 => a22), b2
end

function _interface_closure_two_phase(
	model::TransportModelTwoPhase{N,T},
	κ1::AbstractVector{T},
	κ2::AbstractVector{T},
	t::T,
) where {N,T}
	_ = t
	nt = model.cap1.ntotal
	length(κ1) == nt || throw(DimensionMismatch("phase-1 interface coefficient length must be $nt"))
	length(κ2) == nt || throw(DimensionMismatch("phase-2 interface coefficient length must be $nt"))
	b21 = zeros(T, nt)
	b22 = zeros(T, nt)
	b23 = zeros(T, nt)
	b24 = zeros(T, nt)
	b41 = zeros(T, nt)
	b42 = zeros(T, nt)
	b43 = zeros(T, nt)
	b44 = zeros(T, nt)
	rhs2 = zeros(T, nt)
	rhs4 = zeros(T, nt)

	LI = LinearIndices(model.cap1.nnodes)
	for I in CartesianIndices(model.cap1.nnodes)
		lin = LI[I]
		any(d -> I[d] == model.cap1.nnodes[d], 1:N) && continue

		Γ1 = model.cap1.buf.Γ[lin]
		Γ2 = model.cap2.buf.Γ[lin]
		have_iface = (isfinite(Γ1) && Γ1 > zero(T)) || (isfinite(Γ2) && Γ2 > zero(T))
		have_iface || continue

		tol = convert(T, 100) * eps(T) * max(one(T), abs(Γ1), abs(Γ2))
		abs(Γ1 - Γ2) <= tol || throw(ArgumentError("interface measure mismatch at cell $lin: Γ1=$Γ1 Γ2=$Γ2"))
		Γ = convert(T, 0.5) * (Γ1 + Γ2)
		(isfinite(Γ) && Γ > zero(T)) || continue

		κ1i = κ1[lin]
		κ2i = κ2[lin]
		scale = max(isfinite(κ1i) ? abs(κ1i) : zero(T), isfinite(κ2i) ? abs(κ2i) : zero(T))
		in1 = _is_inflow_coeff(κ1i, scale)
		in2 = _is_inflow_coeff(κ2i, scale)

		if in1 && in2
			throw(ArgumentError("two-phase transport interface is locally ill-posed (both-inflow local configuration) at Γ cell $lin"))
		elseif in1 && !in2
			# gamma1 row: phase-1 inflow receives discrete flux continuity.
			b22[lin] = κ1i
			b24[lin] = κ2i
			b43[lin] = -Γ
			b44[lin] = Γ
		elseif in2 && !in1
			# gamma2 row: phase-2 inflow receives discrete flux continuity.
			b21[lin] = -Γ
			b22[lin] = Γ
			b42[lin] = κ1i
			b44[lin] = κ2i
		else
			# both outflow (or near zero): continuity on each phase.
			b21[lin] = -Γ
			b22[lin] = Γ
			b43[lin] = -Γ
			b44[lin] = Γ
		end
	end

	return (
		spdiagm(0 => b21), spdiagm(0 => b22), spdiagm(0 => b23), spdiagm(0 => b24),
		spdiagm(0 => b41), spdiagm(0 => b42), spdiagm(0 => b43), spdiagm(0 => b44),
		rhs2, rhs4,
	)
end

function _interface_closure_two_phase_moving(
	model::MovingTransportModelTwoPhase{N,T},
	cap1::AssembledCapacity{N,T},
	cap2::AssembledCapacity{N,T},
	κ1rel::AbstractVector{T},
	κ2rel::AbstractVector{T},
	t::T,
) where {N,T}
	_ = t
	nt = cap1.ntotal
	length(κ1rel) == nt || throw(DimensionMismatch("phase-1 relative interface coefficient length must be $nt"))
	length(κ2rel) == nt || throw(DimensionMismatch("phase-2 relative interface coefficient length must be $nt"))
	b21 = zeros(T, nt)
	b22 = zeros(T, nt)
	b23 = zeros(T, nt)
	b24 = zeros(T, nt)
	b41 = zeros(T, nt)
	b42 = zeros(T, nt)
	b43 = zeros(T, nt)
	b44 = zeros(T, nt)
	rhs2 = zeros(T, nt)
	rhs4 = zeros(T, nt)

	LI = LinearIndices(cap1.nnodes)
	for I in CartesianIndices(cap1.nnodes)
		lin = LI[I]
		any(d -> I[d] == cap1.nnodes[d], 1:N) && continue

		Γ1 = cap1.buf.Γ[lin]
		Γ2 = cap2.buf.Γ[lin]
		have_iface = (isfinite(Γ1) && Γ1 > zero(T)) || (isfinite(Γ2) && Γ2 > zero(T))
		have_iface || continue

		tol = convert(T, 100) * eps(T) * max(one(T), abs(Γ1), abs(Γ2))
		abs(Γ1 - Γ2) <= tol || throw(ArgumentError("interface measure mismatch at cell $lin: Γ1=$Γ1 Γ2=$Γ2"))
		Γ = convert(T, 0.5) * (Γ1 + Γ2)
		(isfinite(Γ) && Γ > zero(T)) || continue

		κ1i = κ1rel[lin]
		κ2i = κ2rel[lin]
		scale = max(isfinite(κ1i) ? abs(κ1i) : zero(T), isfinite(κ2i) ? abs(κ2i) : zero(T))
		in1 = _is_inflow_coeff(κ1i, scale)
		in2 = _is_inflow_coeff(κ2i, scale)

		if in1 && in2
			throw(ArgumentError("moving two-phase transport interface is locally ill-posed (both-inflow local configuration from relative speeds) at Γ cell $lin"))
		elseif in1 && !in2
			# gamma1 row: phase-1 relative inflow receives discrete flux continuity.
			b22[lin] = κ1i
			b24[lin] = κ2i
			b43[lin] = -Γ
			b44[lin] = Γ
		elseif in2 && !in1
			# gamma2 row: phase-2 relative inflow receives discrete flux continuity.
			b21[lin] = -Γ
			b22[lin] = Γ
			b42[lin] = κ1i
			b44[lin] = κ2i
		else
			# both relative-outflow (or near zero): continuity on each phase
			b21[lin] = -Γ
			b22[lin] = Γ
			b43[lin] = -Γ
			b44[lin] = Γ
		end
	end

	return (
		spdiagm(0 => b21), spdiagm(0 => b22), spdiagm(0 => b23), spdiagm(0 => b24),
		spdiagm(0 => b41), spdiagm(0 => b42), spdiagm(0 => b43), spdiagm(0 => b44),
		rhs2, rhs4,
	)
end

function _normal_face_point(cap::AssembledCapacity{N,T}, I::CartesianIndex{N}, d::Int, is_high::Bool) where {N,T}
	x_d = is_high ? cap.xyz[d][end] : cap.xyz[d][1]
	lin = LinearIndices(cap.nnodes)[I]
	Cω = cap.C_ω[lin]
	return SVector{N,T}(ntuple(k -> k == d ? x_d : Cω[k], N))
end

function apply_box_bc_transport_mono!(
	A::SparseMatrixCSC{T,Int},
	b::Vector{T},
	cap::AssembledCapacity{N,T},
	uω::NTuple{N,Vector{T}},
	bc_border::BorderConditions,
	adv_scheme::AdvectionScheme;
	t::T=zero(T),
	ωrows::UnitRange{Int}=layout_mono(cap.ntotal).offsets.ω,
) where {N,T}
	validate_borderconditions!(bc_border, N)
	length(ωrows) == cap.ntotal || throw(ArgumentError("ω row range length ($(length(ωrows))) must match cap.ntotal ($(cap.ntotal))"))
	length(b) >= last(ωrows) || throw(ArgumentError("rhs vector does not contain ω block"))

	LI = LinearIndices(cap.nnodes)
	pairs = if N == 1
		((:left, :right),)
	elseif N == 2
		((:left, :right), (:bottom, :top))
	elseif N == 3
		((:left, :right), (:bottom, :top), (:backward, :forward))
	else
		throw(ArgumentError("unsupported dimension N=$N; expected 1,2,3"))
	end

	for pair in pairs
		for side in pair
			side_bc = get(bc_border.borders, side, Outflow())
			side_bc isa Periodic && continue
			side_bc isa Outflow && continue
			side_bc isa Inflow || throw(ArgumentError("unsupported advection outer boundary type $(typeof(side_bc))"))

			d, is_high, normal_sign = side_info(side, N)
			for I in each_boundary_cell(cap.nnodes, side)
				row_lin = LI[I]
				row = ωrows[row_lin]
				Aface = cap.buf.A[d][row_lin]
				if !(isfinite(Aface) && Aface != zero(T))
					continue
				end

				un = convert(T, normal_sign) * uω[d][row_lin]
				un < zero(T) || continue

				xface = _normal_face_point(cap, I, d, is_high)
				g = convert(T, eval_bc(side_bc.value, xface, t))

				if adv_scheme isa Centered
					b[row] -= (un * Aface) * (g / convert(T, 2))
				elseif adv_scheme isa Upwind1
					b[row] -= (un * Aface) * g
				else
					throw(ArgumentError("unknown advection scheme $(typeof(adv_scheme))"))
				end
			end
		end
	end

	return A, b
end

function _is_canonical_mono_layout(lay, nt::Int)
	return lay.ω == (1:nt) && lay.γ == ((nt + 1):(2 * nt))
end

function _ops_for_time(
	cap::AssembledCapacity{N,T},
	uω,
	uγ,
	periodic::NTuple{N,Bool},
	scheme::AdvectionScheme,
	t::T,
) where {N,T}
	uωv, uγv = _velocity_values(cap, uω, uγ, t)
	ops = _advection_ops_moving(cap, uωv, uγv; periodic=periodic, scheme=scheme)
	return ops, uωv, uγv
end

function _ops_for_time(model::TransportModelMono{N,T}, t::T) where {N,T}
	ops, uωv, uγv = _ops_for_time(model.cap, model.uω, model.uγ, model.periodic, model.scheme, t)
	return ops, uωv, uγv
end

"""
    update_advection_ops!(model; t=0)

Rebuild and store advection operators at time `t` for a mono or two-phase model.
"""
function update_advection_ops!(model::TransportModelMono{N,T}; t::T=zero(T)) where {N,T}
	ops, _, _ = _ops_for_time(model, t)
	model.ops = ops
	return model
end

function update_advection_ops!(model::TransportModelTwoPhase{N,T}; t::T=zero(T)) where {N,T}
	ops1, _, _ = _ops_for_time(model.cap1, model.u1ω, model.u1γ, model.periodic1, model.scheme, t)
	ops2, _, _ = _ops_for_time(model.cap2, model.u2ω, model.u2γ, model.periodic2, model.scheme, t)
	model.ops1 = ops1
	model.ops2 = ops2
	return model
end

"""
    rebuild!(model::TransportModelMono, moments; bc=0, t=0)

Rebuild capacity geometry and refresh advection operators for a monophasic model.
"""
function rebuild!(model::TransportModelMono{N,T}, moments; bc=zero(T), t::T=zero(T)) where {N,T}
	CartesianOperators.rebuild!(model.cap, moments; bc=bc)
	update_advection_ops!(model; t=t)
	return model
end

"""
    assemble_steady_mono!(sys, model, t)

Assemble the steady monophasic linear system at time `t`.
"""
function assemble_steady_mono!(sys::LinearSystem{T}, model::TransportModelMono{N,T}, t::T) where {N,T}
	nt = model.cap.ntotal
	lay = model.layout.offsets
	_validate_mono_layout(model.cap, lay)
	nsys = maximum((last(lay.ω), last(lay.γ)))

	ops, uωv, _ = _ops_for_time(model, t)
	model.ops = ops

	# Transport assembly mirrors the diffusion block structure:
	# - ops.C: bulk face transport on ω rows
	# - ops.K: embedded-interface transport contribution to ω rows
	# γ rows remain closure/transmission equations for interface unknowns.
	# The advection scheme only changes how κ is split between Tω and Tγ.
	conv_bulk = reduce(+, ops.C)
	κ = _interface_flux_diag(ops)
	κω, κγ = _split_interface_flux(κ, model.scheme)
	A11 = conv_bulk + spdiagm(0 => κω)
	A12 = spdiagm(0 => κγ)

	fω = _source_values(model.cap, model.source, t)
	b1 = model.cap.V * fω
	A21, A22, b2 = _interface_closure(model, κ, t)

	A, b = if _is_canonical_mono_layout(lay, nt)
		([A11 A12; A21 A22], vcat(b1, b2))
	else
		Awork = spzeros(T, nsys, nsys)
		bwork = zeros(T, nsys)
		_insert_block!(Awork, lay.ω, lay.ω, A11)
		_insert_block!(Awork, lay.ω, lay.γ, A12)
		_insert_block!(Awork, lay.γ, lay.ω, A21)
		_insert_block!(Awork, lay.γ, lay.γ, A22)
		_insert_vec!(bwork, lay.ω, b1)
		_insert_vec!(bwork, lay.γ, b2)
		(Awork, bwork)
	end

	sys.A = A
	sys.b = b
	length(sys.x) == nsys || (sys.x = zeros(T, nsys))
	sys.cache = nothing

	apply_box_bc_transport_mono!(sys.A, sys.b, model.cap, uωv, model.bc_border, model.scheme; t=t, ωrows=model.layout.offsets.ω)
	active_rows = _mono_row_activity(model.cap, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
	return sys
end

"""
    assemble_steady_two_phase!(sys, model, t)

Assemble the steady two-phase linear system at time `t` with interface flux coupling.
"""
function assemble_steady_two_phase!(sys::LinearSystem{T}, model::TransportModelTwoPhase{N,T}, t::T) where {N,T}
	_validate_two_phase_caps(model.cap1, model.cap2)
	lay = model.layout
	_validate_two_phase_layout(model.cap1, model.cap2, lay)
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

	ops1, u1ωv, _ = _ops_for_time(model.cap1, model.u1ω, model.u1γ, model.periodic1, model.scheme, t)
	ops2, u2ωv, _ = _ops_for_time(model.cap2, model.u2ω, model.u2γ, model.periodic2, model.scheme, t)
	model.ops1 = ops1
	model.ops2 = ops2

	conv_bulk1 = reduce(+, ops1.C)
	κ1 = _interface_flux_diag(ops1)
	κ1ω, κ1γ = _split_interface_flux(κ1, model.scheme)
	A11 = conv_bulk1 + spdiagm(0 => κ1ω)
	A12 = spdiagm(0 => κ1γ)
	f1 = _source_values(model.cap1, model.source1, t)
	b1 = model.cap1.V * f1

	conv_bulk2 = reduce(+, ops2.C)
	κ2 = _interface_flux_diag(ops2)
	κ2ω, κ2γ = _split_interface_flux(κ2, model.scheme)
	A33 = conv_bulk2 + spdiagm(0 => κ2ω)
	A34 = spdiagm(0 => κ2γ)
	f2 = _source_values(model.cap2, model.source2, t)
	b3 = model.cap2.V * f2

	B21, B22, B23, B24, B41, B42, B43, B44, b2, b4 = _interface_closure_two_phase(model, κ1, κ2, t)

	A = spzeros(T, nsys, nsys)
	b = zeros(T, nsys)
	_insert_block!(A, lay.ω1, lay.ω1, A11)
	_insert_block!(A, lay.ω1, lay.γ1, A12)
	_insert_block!(A, lay.γ1, lay.ω1, B21)
	_insert_block!(A, lay.γ1, lay.γ1, B22)
	_insert_block!(A, lay.γ1, lay.ω2, B23)
	_insert_block!(A, lay.γ1, lay.γ2, B24)
	_insert_block!(A, lay.ω2, lay.ω2, A33)
	_insert_block!(A, lay.ω2, lay.γ2, A34)
	_insert_block!(A, lay.γ2, lay.ω1, B41)
	_insert_block!(A, lay.γ2, lay.γ1, B42)
	_insert_block!(A, lay.γ2, lay.ω2, B43)
	_insert_block!(A, lay.γ2, lay.γ2, B44)
	_insert_vec!(b, lay.ω1, b1)
	_insert_vec!(b, lay.γ1, b2)
	_insert_vec!(b, lay.ω2, b3)
	_insert_vec!(b, lay.γ2, b4)

	sys.A = A
	sys.b = b
	length(sys.x) == nsys || (sys.x = zeros(T, nsys))
	sys.cache = nothing

	apply_box_bc_transport_mono!(sys.A, sys.b, model.cap1, u1ωv, model.bc_border1, model.scheme; t=t, ωrows=lay.ω1)
	apply_box_bc_transport_mono!(sys.A, sys.b, model.cap2, u2ωv, model.bc_border2, model.scheme; t=t, ωrows=lay.ω2)

	active_rows = _two_phase_row_activity(model.cap1, model.cap2, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
	return sys
end

function _resolve_theta(scheme)::Float64
	if scheme === :BE
		return 1.0
	elseif scheme === :CN
		return 0.5
	elseif scheme isa Symbol
		throw(ArgumentError("scheme must be :BE, :CN, or a numeric theta in [0,1]; got Symbol `$scheme`"))
	elseif scheme isa Bool
		throw(ArgumentError("scheme must be :BE, :CN, or a numeric theta in [0,1]"))
	elseif scheme isa Real
		θ = try
			Float64(scheme)
		catch
			throw(ArgumentError("scheme must be :BE, :CN, or a numeric theta in [0,1]"))
		end
		(isfinite(θ) && 0.0 <= θ <= 1.0) || throw(ArgumentError("scheme must be :BE, :CN, or a numeric theta in [0,1]"))
		return θ
	end
	throw(ArgumentError("scheme must be :BE, :CN, or a numeric theta in [0,1]"))
end

function _psi_functions(::Type{T}, θ::T) where {T}
	(zero(T) <= θ <= one(T)) || throw(ArgumentError("numeric θ must satisfy 0 ≤ θ ≤ 1"))
	psip = (Vn, Vn1) -> begin
		if iszero(Vn) && iszero(Vn1)
			return zero(T)
		elseif iszero(Vn) && Vn1 > zero(T)
			return one(T) # fresh cell: implicit-only update
		elseif Vn > zero(T) && iszero(Vn1)
			return zero(T) # dead cell: no new-state contribution
		end
		return θ
	end
	psim = (Vn, Vn1) -> begin
		if iszero(Vn) && iszero(Vn1)
			return zero(T)
		elseif iszero(Vn) && Vn1 > zero(T)
			return zero(T) # fresh cell: no old-state explicit contribution
		elseif Vn > zero(T) && iszero(Vn1)
			return one(T) # dead cell: old-state-only contribution
		end
		return one(T) - θ
	end
	return psip, psim
end

function _init_unsteady_state_mono(model::TransportModelMono{N,T}, u0) where {N,T}
	lay = model.layout.offsets
	nt = model.cap.ntotal
	nsys = maximum((last(lay.ω), last(lay.γ)))
	u = zeros(T, nsys)
	if length(u0) == nsys
		u .= Vector{T}(u0)
	elseif length(u0) == nt
		u[lay.ω] .= Vector{T}(u0)
	else
		throw(DimensionMismatch("u0 length must be $nt (ω block) or $nsys (full system)"))
	end
	return u
end

function _as_full_state_two_phase(model::TransportModelTwoPhase{N,T}, u0) where {N,T}
	lay = model.layout
	nt = model.cap1.ntotal
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
	u = zeros(T, nsys)

	if u0 isa Tuple
		length(u0) == 2 || throw(DimensionMismatch("tuple initial state must be (u01, u02)"))
		length(u0[1]) == nt || throw(DimensionMismatch("u01 length must be $nt"))
		length(u0[2]) == nt || throw(DimensionMismatch("u02 length must be $nt"))
		u[lay.ω1] .= Vector{T}(u0[1])
		u[lay.ω2] .= Vector{T}(u0[2])
		return u
	end

	len = length(u0)
	if len == nsys
		u .= Vector{T}(u0)
	elseif len == 2 * nt
		u[lay.ω1] .= Vector{T}(u0[1:nt])
		u[lay.ω2] .= Vector{T}(u0[(nt + 1):(2 * nt)])
	else
		throw(DimensionMismatch("u0 must be length $nsys (full state), length $(2 * nt) (ω blocks concatenated), or tuple (u01, u02)"))
	end
	return u
end

function _init_unsteady_state_two_phase(model::TransportModelTwoPhase{N,T}, u0) where {N,T}
	return _as_full_state_two_phase(model, u0)
end

function _as_full_state_two_phase_moving(model::MovingTransportModelTwoPhase{N,T}, u0) where {N,T}
	lay = model.layout
	nt = prod(model.grid.n)
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
	u = zeros(T, nsys)

	if u0 isa Tuple
		length(u0) == 2 || throw(DimensionMismatch("tuple initial state must be (u01, u02)"))
		length(u0[1]) == nt || throw(DimensionMismatch("u01 length must be $nt"))
		length(u0[2]) == nt || throw(DimensionMismatch("u02 length must be $nt"))
		u[lay.ω1] .= Vector{T}(u0[1])
		u[lay.ω2] .= Vector{T}(u0[2])
		return u
	end

	len = length(u0)
	if len == nsys
		u .= Vector{T}(u0)
	elseif len == 2 * nt
		u[lay.ω1] .= Vector{T}(u0[1:nt])
		u[lay.ω2] .= Vector{T}(u0[(nt + 1):(2 * nt)])
	else
		throw(DimensionMismatch("u0 must be length $nsys (full state), length $(2 * nt) (ω blocks concatenated), or tuple (u01, u02)"))
	end
	return u
end

function _init_unsteady_state_moving(model::MovingTransportModelMono{N,T}, u0) where {N,T}
	lay = model.layout.offsets
	nt = prod(model.grid.n)
	nsys = maximum((last(lay.ω), last(lay.γ)))
	u = zeros(T, nsys)
	if length(u0) == nsys
		u .= Vector{T}(u0)
	elseif length(u0) == nt
		u[lay.ω] .= Vector{T}(u0)
	else
		throw(DimensionMismatch("u0 length must be $nt (ω block) or $nsys (full system)"))
	end
	return u
end

function _init_unsteady_state_moving(model::MovingTransportModelTwoPhase{N,T}, u0) where {N,T}
	return _as_full_state_two_phase_moving(model, u0)
end

"""
    assemble_unsteady_mono!(sys, model, uⁿ, t, dt, scheme_or_theta)

Assemble the monophasic theta-method system for one unsteady step.
"""
function assemble_unsteady_mono!(
	sys::LinearSystem{T},
	model::TransportModelMono{N,T},
	uⁿ,
	t::T,
	dt::T,
	scheme_or_theta,
) where {N,T}
	θ = convert(T, _resolve_theta(scheme_or_theta))
	assemble_steady_mono!(sys, model, t + θ * dt)

	lay = model.layout.offsets
	_validate_mono_layout(model.cap, lay)
	nt = model.cap.ntotal
	nsys = maximum((last(lay.ω), last(lay.γ)))

	ufull = if length(uⁿ) == nsys
		Vector{T}(uⁿ)
	elseif length(uⁿ) == nt
		v = zeros(T, nsys)
		v[lay.ω] .= Vector{T}(uⁿ)
		v
	else
		v = zeros(T, nsys)
		v[lay.ω] .= Vector{T}(uⁿ[lay.ω])
		v
	end

	if θ != one(T)
		Aω_prev = sys.A[lay.ω, :]
		corr = Aω_prev * ufull
		_scale_rows!(sys.A, lay.ω, θ)
		_insert_vec!(sys.b, lay.ω, (-(one(T) - θ)) .* corr)
	end

	M = model.cap.buf.V ./ dt
	sys.A = sys.A + sparse(lay.ω, lay.ω, M, nsys, nsys)
	_insert_vec!(sys.b, lay.ω, M .* Vector{T}(ufull[lay.ω]))

	active_rows = _mono_row_activity(model.cap, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
	sys.cache = nothing
	return sys
end

"""
    assemble_unsteady_two_phase!(sys, model, uⁿ, t, dt, scheme_or_theta)

Assemble the two-phase theta-method system for one unsteady step.
"""
function assemble_unsteady_two_phase!(
	sys::LinearSystem{T},
	model::TransportModelTwoPhase{N,T},
	uⁿ,
	t::T,
	dt::T,
	scheme_or_theta,
) where {N,T}
	θ = convert(T, _resolve_theta(scheme_or_theta))
	assemble_steady_two_phase!(sys, model, t + θ * dt)

	lay = model.layout
	_validate_two_phase_layout(model.cap1, model.cap2, lay)
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
	ufull = _as_full_state_two_phase(model, uⁿ)

	if θ != one(T)
		Aω1_prev = sys.A[lay.ω1, :]
		Aω2_prev = sys.A[lay.ω2, :]
		corr1 = Aω1_prev * ufull
		corr2 = Aω2_prev * ufull
		_scale_rows!(sys.A, lay.ω1, θ)
		_scale_rows!(sys.A, lay.ω2, θ)
		_insert_vec!(sys.b, lay.ω1, (-(one(T) - θ)) .* corr1)
		_insert_vec!(sys.b, lay.ω2, (-(one(T) - θ)) .* corr2)
	end

	M1 = model.cap1.buf.V ./ dt
	M2 = model.cap2.buf.V ./ dt
	sys.A = sys.A + sparse(lay.ω1, lay.ω1, M1, nsys, nsys)
	sys.A = sys.A + sparse(lay.ω2, lay.ω2, M2, nsys, nsys)
	_insert_vec!(sys.b, lay.ω1, M1 .* Vector{T}(ufull[lay.ω1]))
	_insert_vec!(sys.b, lay.ω2, M2 .* Vector{T}(ufull[lay.ω2]))

	active_rows = _two_phase_row_activity(model.cap1, model.cap2, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
	sys.cache = nothing
	return sys
end

"""
    assemble_unsteady_mono_moving!(sys, model, uⁿ, t, dt, scheme_or_theta)

Assemble one moving-geometry monophasic theta-method step on slab `[t, t+dt]`.
Embedded-interface inflow/outflow switching uses relative speed `(uγ - wγ)·nγ`.
"""
function assemble_unsteady_mono_moving!(
	sys::LinearSystem{T},
	model::MovingTransportModelMono{N,T},
	uⁿ,
	t::T,
	dt::T,
	scheme_or_theta,
) where {N,T}
	dt > zero(T) || throw(ArgumentError("dt must be positive"))
	θ = convert(T, _resolve_theta(scheme_or_theta))
	psip, psim = _psi_functions(T, θ)
	tθ = t + θ * dt

	_build_moving_slab!(model, t, dt)
	cap = model.cap_slab
	cap === nothing && throw(ArgumentError("missing slab capacity cache"))
	uωv, uγv = _velocity_values(cap, model.uω, model.uγ, tθ)
	wγv = _velocity_tuple_values(cap, model.wγ, cap.C_γ, tθ)
	uγrel = _relative_interface_velocity(uγv, wγv)
	ops = _advection_ops_moving(cap, uωv, uγrel; periodic=model.periodic, scheme=model.scheme)
	model.ops_slab = ops

	lay = model.layout.offsets
	_validate_mono_layout(cap, lay)
	nt = cap.ntotal
	nsys = maximum((last(lay.ω), last(lay.γ)))

	ufull = if length(uⁿ) == nsys
		Vector{T}(uⁿ)
	elseif length(uⁿ) == nt
		v = zeros(T, nsys)
		v[lay.ω] .= Vector{T}(uⁿ)
		v
	else
		v = zeros(T, nsys)
		v[lay.ω] .= Vector{T}(uⁿ[lay.ω])
		if length(uⁿ) >= last(lay.γ)
			v[lay.γ] .= Vector{T}(uⁿ[lay.γ])
		end
		v
	end

	conv_bulk = reduce(+, ops.C)
	κrel = _interface_flux_diag(ops)
	κω, κγ = _split_interface_flux(κrel, model.scheme)
	Adv11 = conv_bulk + spdiagm(0 => κω)
	Adv12 = spdiagm(0 => κγ)
	fω = _source_values(cap, model.source, tθ)
	A21, A22, b2 = _interface_closure_moving(model, cap, κrel, tθ)

	M1 = spdiagm(0 => model.Vn1)
	M0 = spdiagm(0 => model.Vn)
	Ψp = spdiagm(0 => T[psip(model.Vn[i], model.Vn1[i]) for i in 1:nt])
	Ψm = spdiagm(0 => T[psim(model.Vn[i], model.Vn1[i]) for i in 1:nt])

	# Geometric sweep is represented by Vn1-Vn; no standalone geometry term is added.
	A11 = M1 + Adv11 * Ψp
	A12 = -(M1 - M0) + Adv12 * Ψp

	uω = Vector{T}(ufull[lay.ω])
	uγ = Vector{T}(ufull[lay.γ])
	b1 = (M0 - Adv11 * Ψm) * uω
	b1 .-= (Adv12 * Ψm) * uγ
	b1 .+= cap.V * fω

	A, b = if _is_canonical_mono_layout(lay, nt)
		([A11 A12; A21 A22], vcat(b1, b2))
	else
		Awork = spzeros(T, nsys, nsys)
		bwork = zeros(T, nsys)
		_insert_block!(Awork, lay.ω, lay.ω, A11)
		_insert_block!(Awork, lay.ω, lay.γ, A12)
		_insert_block!(Awork, lay.γ, lay.ω, A21)
		_insert_block!(Awork, lay.γ, lay.γ, A22)
		_insert_vec!(bwork, lay.ω, b1)
		_insert_vec!(bwork, lay.γ, b2)
		(Awork, bwork)
	end

	sys.A = A
	sys.b = b
	length(sys.x) == nsys || (sys.x = zeros(T, nsys))
	sys.cache = nothing

	apply_box_bc_transport_mono!(sys.A, sys.b, cap, uωv, model.bc_border, model.scheme; t=tθ, ωrows=lay.ω)
	active_rows = _mono_row_activity(cap, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)

	sys.cache = nothing
	return sys
end

"""
    assemble_unsteady_two_phase_moving!(sys, model, uⁿ, t, dt, scheme_or_theta)

Assemble one moving-geometry two-phase theta-method step on slab `[t, t+dt]`.
Embedded-interface inflow/outflow switching uses relative speeds against `wγ`.
"""
function assemble_unsteady_two_phase_moving!(
	sys::LinearSystem{T},
	model::MovingTransportModelTwoPhase{N,T},
	uⁿ,
	t::T,
	dt::T,
	scheme_or_theta,
) where {N,T}
	dt > zero(T) || throw(ArgumentError("dt must be positive"))
	θ = convert(T, _resolve_theta(scheme_or_theta))
	psip, psim = _psi_functions(T, θ)
	tθ = t + θ * dt

	_build_moving_slab!(model, t, dt)
	cap1 = model.cap1_slab
	cap2 = model.cap2_slab
	cap1 === nothing && throw(ArgumentError("missing phase-1 slab capacity cache"))
	cap2 === nothing && throw(ArgumentError("missing phase-2 slab capacity cache"))
	_validate_two_phase_caps(cap1, cap2)

	u1ωv, u1γv = _velocity_values(cap1, model.u1ω, model.u1γ, tθ)
	u2ωv, u2γv = _velocity_values(cap2, model.u2ω, model.u2γ, tθ)
	wγv = _velocity_tuple_values(cap1, model.wγ, cap1.C_γ, tθ)
	u1γrel = _relative_interface_velocity(u1γv, wγv)
	u2γrel = _relative_interface_velocity(u2γv, wγv)
	ops1 = _advection_ops_moving(cap1, u1ωv, u1γrel; periodic=model.periodic1, scheme=model.scheme)
	ops2 = _advection_ops_moving(cap2, u2ωv, u2γrel; periodic=model.periodic2, scheme=model.scheme)
	model.ops1_slab = ops1
	model.ops2_slab = ops2

	lay = model.layout
	_validate_two_phase_layout(cap1, cap2, lay)
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
	nt = cap1.ntotal
	ufull = _as_full_state_two_phase_moving(model, uⁿ)

	conv_bulk1 = reduce(+, ops1.C)
	κ1rel = _interface_flux_diag(ops1)
	κ1ω, κ1γ = _split_interface_flux(κ1rel, model.scheme)
	Adv11 = conv_bulk1 + spdiagm(0 => κ1ω)
	Adv12 = spdiagm(0 => κ1γ)
	f1 = _source_values(cap1, model.source1, tθ)

	conv_bulk2 = reduce(+, ops2.C)
	κ2rel = _interface_flux_diag(ops2)
	κ2ω, κ2γ = _split_interface_flux(κ2rel, model.scheme)
	Adv33 = conv_bulk2 + spdiagm(0 => κ2ω)
	Adv34 = spdiagm(0 => κ2γ)
	f2 = _source_values(cap2, model.source2, tθ)

	B21, B22, B23, B24, B41, B42, B43, B44, b2, b4 =
		_interface_closure_two_phase_moving(model, cap1, cap2, κ1rel, κ2rel, tθ)

	M1n1 = spdiagm(0 => model.V1n1)
	M1n = spdiagm(0 => model.V1n)
	M2n1 = spdiagm(0 => model.V2n1)
	M2n = spdiagm(0 => model.V2n)
	Ψ1p = spdiagm(0 => T[psip(model.V1n[i], model.V1n1[i]) for i in 1:nt])
	Ψ1m = spdiagm(0 => T[psim(model.V1n[i], model.V1n1[i]) for i in 1:nt])
	Ψ2p = spdiagm(0 => T[psip(model.V2n[i], model.V2n1[i]) for i in 1:nt])
	Ψ2m = spdiagm(0 => T[psim(model.V2n[i], model.V2n1[i]) for i in 1:nt])

	# Geometric sweep is represented by Vn1-Vn; no standalone geometry term is added.
	A11 = M1n1 + Adv11 * Ψ1p
	A12 = -(M1n1 - M1n) + Adv12 * Ψ1p
	A33 = M2n1 + Adv33 * Ψ2p
	A34 = -(M2n1 - M2n) + Adv34 * Ψ2p

	uω1 = Vector{T}(ufull[lay.ω1])
	uγ1 = Vector{T}(ufull[lay.γ1])
	uω2 = Vector{T}(ufull[lay.ω2])
	uγ2 = Vector{T}(ufull[lay.γ2])
	b1 = (M1n - Adv11 * Ψ1m) * uω1
	b1 .-= (Adv12 * Ψ1m) * uγ1
	b1 .+= cap1.V * f1
	b3 = (M2n - Adv33 * Ψ2m) * uω2
	b3 .-= (Adv34 * Ψ2m) * uγ2
	b3 .+= cap2.V * f2

	A = spzeros(T, nsys, nsys)
	b = zeros(T, nsys)
	_insert_block!(A, lay.ω1, lay.ω1, A11)
	_insert_block!(A, lay.ω1, lay.γ1, A12)
	_insert_block!(A, lay.γ1, lay.ω1, B21)
	_insert_block!(A, lay.γ1, lay.γ1, B22)
	_insert_block!(A, lay.γ1, lay.ω2, B23)
	_insert_block!(A, lay.γ1, lay.γ2, B24)
	_insert_block!(A, lay.ω2, lay.ω2, A33)
	_insert_block!(A, lay.ω2, lay.γ2, A34)
	_insert_block!(A, lay.γ2, lay.ω1, B41)
	_insert_block!(A, lay.γ2, lay.γ1, B42)
	_insert_block!(A, lay.γ2, lay.ω2, B43)
	_insert_block!(A, lay.γ2, lay.γ2, B44)
	_insert_vec!(b, lay.ω1, b1)
	_insert_vec!(b, lay.γ1, b2)
	_insert_vec!(b, lay.ω2, b3)
	_insert_vec!(b, lay.γ2, b4)

	sys.A = A
	sys.b = b
	length(sys.x) == nsys || (sys.x = zeros(T, nsys))
	sys.cache = nothing

	apply_box_bc_transport_mono!(sys.A, sys.b, cap1, u1ωv, model.bc_border1, model.scheme; t=tθ, ωrows=lay.ω1)
	apply_box_bc_transport_mono!(sys.A, sys.b, cap2, u2ωv, model.bc_border2, model.scheme; t=tθ, ωrows=lay.ω2)
	active_rows = _two_phase_row_activity(cap1, cap2, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
	sys.cache = nothing
	return sys
end

function PenguinSolverCore.assemble!(sys::LinearSystem{T}, model::TransportModelMono{N,T}, t, dt) where {N,T}
	assemble_unsteady_mono!(sys, model, sys.x, convert(T, t), convert(T, dt), one(T))
end

function PenguinSolverCore.assemble!(sys::LinearSystem{T}, model::TransportModelTwoPhase{N,T}, t, dt) where {N,T}
	assemble_unsteady_two_phase!(sys, model, sys.x, convert(T, t), convert(T, dt), one(T))
end

"""
    solve_steady!(model; t=0, method=:direct, kwargs...)

Assemble and solve the steady linear system for mono or two-phase transport models.
"""
function solve_steady!(model::TransportModelMono{N,T}; t::T=zero(T), method::Symbol=:direct, kwargs...) where {N,T}
	n = maximum((last(model.layout.offsets.ω), last(model.layout.offsets.γ)))
	sys = LinearSystem(spzeros(T, n, n), zeros(T, n))
	assemble_steady_mono!(sys, model, t)
	solve!(sys; method=method, kwargs...)
	return sys
end

function solve_steady!(model::TransportModelTwoPhase{N,T}; t::T=zero(T), method::Symbol=:direct, kwargs...) where {N,T}
	n = maximum((last(model.layout.ω1), last(model.layout.γ1), last(model.layout.ω2), last(model.layout.γ2)))
	sys = LinearSystem(spzeros(T, n, n), zeros(T, n))
	assemble_steady_two_phase!(sys, model, t)
	solve!(sys; method=method, kwargs...)
	return sys
end

function _source_time_dependent(source, x::SVector{N,T}) where {N,T}
	return source isa Function && applicable(source, x..., zero(T))
end

function _matrix_time_dependent(model::TransportModelMono{N,T}) where {N,T}
	xω = model.cap.C_ω[1]
	xγ = model.cap.C_γ[1]
	return _vel_time_dependent(model.uω, xω) || _vel_time_dependent(model.uγ, xγ)
end

function _rhs_time_dependent(model::TransportModelMono{N,T}) where {N,T}
	xω = model.cap.C_ω[1]
	xγ = model.cap.C_γ[1]
	return _source_time_dependent(model.source, xω) ||
		_adv_border_time_dependent(model.bc_border, xω) ||
		_interface_time_dependent(model.bc_interface, xγ) ||
		_vel_time_dependent(model.uω, xω) ||
		_vel_time_dependent(model.uγ, xγ)
end

function _matrix_time_dependent(model::TransportModelTwoPhase{N,T}) where {N,T}
	x1ω = model.cap1.C_ω[1]
	x1γ = model.cap1.C_γ[1]
	x2ω = model.cap2.C_ω[1]
	x2γ = model.cap2.C_γ[1]
	return _vel_time_dependent(model.u1ω, x1ω) ||
		_vel_time_dependent(model.u1γ, x1γ) ||
		_vel_time_dependent(model.u2ω, x2ω) ||
		_vel_time_dependent(model.u2γ, x2γ)
end

function _rhs_time_dependent(model::TransportModelTwoPhase{N,T}) where {N,T}
	x1ω = model.cap1.C_ω[1]
	x2ω = model.cap2.C_ω[1]
	return _source_time_dependent(model.source1, x1ω) ||
		_source_time_dependent(model.source2, x2ω) ||
		_adv_border_time_dependent(model.bc_border1, x1ω) ||
		_adv_border_time_dependent(model.bc_border2, x2ω) ||
		_vel_time_dependent(model.u1ω, x1ω) ||
		_vel_time_dependent(model.u1γ, model.cap1.C_γ[1]) ||
		_vel_time_dependent(model.u2ω, x2ω) ||
		_vel_time_dependent(model.u2γ, model.cap2.C_γ[1])
end

function _prepare_constant_unsteady_mono(model::TransportModelMono{N,T}, t0::T, dt::T, θ::T) where {N,T}
	lay = model.layout.offsets
	nsys = maximum((last(lay.ω), last(lay.γ)))
	sys0 = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
	assemble_steady_mono!(sys0, model, t0 + θ * dt)
	Asteady = sys0.A
	bsteady = copy(sys0.b)
	Aconst = copy(Asteady)
	θ != one(T) && _scale_rows!(Aconst, lay.ω, θ)
	M = model.cap.buf.V ./ dt
	Aconst = Aconst + sparse(lay.ω, lay.ω, M, nsys, nsys)
	Aω_prev = Asteady[lay.ω, :]
	return Aconst, bsteady, Aω_prev, M
end

function _set_constant_rhs_mono!(
	b::Vector{T},
	bsteady::Vector{T},
	Aω_prev::SparseMatrixCSC{T,Int},
	M::Vector{T},
	lay,
	u::Vector{T},
	θ::T,
) where {T}
	copyto!(b, bsteady)
	if θ != one(T)
		corr = Aω_prev * u
		_insert_vec!(b, lay.ω, (-(one(T) - θ)) .* corr)
	end
	_insert_vec!(b, lay.ω, M .* Vector{T}(u[lay.ω]))
	return b
end

function _prepare_constant_unsteady_two_phase(model::TransportModelTwoPhase{N,T}, t0::T, dt::T, θ::T) where {N,T}
	lay = model.layout
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
	sys0 = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
	assemble_steady_two_phase!(sys0, model, t0 + θ * dt)
	Asteady = sys0.A
	bsteady = copy(sys0.b)
	Aconst = copy(Asteady)
	if θ != one(T)
		_scale_rows!(Aconst, lay.ω1, θ)
		_scale_rows!(Aconst, lay.ω2, θ)
	end
	M1 = model.cap1.buf.V ./ dt
	M2 = model.cap2.buf.V ./ dt
	Aconst = Aconst + sparse(lay.ω1, lay.ω1, M1, nsys, nsys)
	Aconst = Aconst + sparse(lay.ω2, lay.ω2, M2, nsys, nsys)
	Aω1_prev = Asteady[lay.ω1, :]
	Aω2_prev = Asteady[lay.ω2, :]
	return Aconst, bsteady, Aω1_prev, Aω2_prev, M1, M2
end

function _set_constant_rhs_two_phase!(
	b::Vector{T},
	bsteady::Vector{T},
	Aω1_prev::SparseMatrixCSC{T,Int},
	Aω2_prev::SparseMatrixCSC{T,Int},
	M1::Vector{T},
	M2::Vector{T},
	lay,
	u::Vector{T},
	θ::T,
) where {T}
	copyto!(b, bsteady)
	if θ != one(T)
		corr1 = Aω1_prev * u
		corr2 = Aω2_prev * u
		_insert_vec!(b, lay.ω1, (-(one(T) - θ)) .* corr1)
		_insert_vec!(b, lay.ω2, (-(one(T) - θ)) .* corr2)
	end
	_insert_vec!(b, lay.ω1, M1 .* Vector{T}(u[lay.ω1]))
	_insert_vec!(b, lay.ω2, M2 .* Vector{T}(u[lay.ω2]))
	return b
end

"""
    solve_unsteady!(model, u0, tspan; dt, scheme=:BE, method=:direct, save_history=true, kwargs...)

Time-integrate mono or two-phase transport with a theta-method (`:BE`, `:CN`, or numeric `theta` in `[0,1]`).
Returns `(times, states, system, reused_constant_operator)`.
"""
function solve_unsteady!(
	model::TransportModelMono{N,T},
	u0,
	tspan::Tuple{T,T};
	dt::T,
	scheme=:BE,
	method::Symbol=:direct,
	save_history::Bool=true,
	kwargs...,
) where {N,T}
	t0, tend = tspan
	tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
	dt > zero(T) || throw(ArgumentError("dt must be positive"))
	θ = convert(T, _resolve_theta(scheme))

	u = _init_unsteady_state_mono(model, u0)
	lay = model.layout.offsets
	nsys = maximum((last(lay.ω), last(lay.γ)))

	matrix_dep = _matrix_time_dependent(model)
	rhs_dep = _rhs_time_dependent(model)
	constant_operator = !matrix_dep && !rhs_dep

	times = T[t0]
	states = Vector{Vector{T}}()
	save_history && push!(states, copy(u))

	tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
	t = t0

	if constant_operator
		Aconst, bsteady, Aω_prev, M = _prepare_constant_unsteady_mono(model, t0, dt, θ)
		sys = LinearSystem(Aconst, copy(bsteady); x=copy(u))
		while t < tend - tol
			dt_step = min(dt, tend - t)
			if abs(dt_step - dt) <= tol
				_set_constant_rhs_mono!(sys.b, bsteady, Aω_prev, M, lay, u, θ)
				solve!(sys; method=method, reuse_factorization=true, kwargs...)
			else
				assemble_unsteady_mono!(sys, model, u, t, dt_step, θ)
				solve!(sys; method=method, reuse_factorization=false, kwargs...)
			end
			u .= sys.x
			t += dt_step
			push!(times, t)
			save_history && push!(states, copy(u))
		end
		if !save_history
			states = [copy(u)]
			times = T[t]
		end
		return (times=times, states=states, system=sys, reused_constant_operator=true)
	end

	sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
	while t < tend - tol
		dt_step = min(dt, tend - t)
		assemble_unsteady_mono!(sys, model, u, t, dt_step, θ)
		solve!(sys; method=method, reuse_factorization=false, kwargs...)
		u .= sys.x
		t += dt_step
		push!(times, t)
		save_history && push!(states, copy(u))
	end
	if !save_history
		states = [copy(u)]
		times = T[t]
	end
	return (times=times, states=states, system=sys, reused_constant_operator=false)
end

function solve_unsteady!(
	model::TransportModelTwoPhase{N,T},
	u0,
	tspan::Tuple{T,T};
	dt::T,
	scheme=:BE,
	method::Symbol=:direct,
	save_history::Bool=true,
	kwargs...,
) where {N,T}
	t0, tend = tspan
	tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
	dt > zero(T) || throw(ArgumentError("dt must be positive"))
	θ = convert(T, _resolve_theta(scheme))

	u = _init_unsteady_state_two_phase(model, u0)
	lay = model.layout
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

	matrix_dep = _matrix_time_dependent(model)
	rhs_dep = _rhs_time_dependent(model)
	constant_operator = !matrix_dep && !rhs_dep

	times = T[t0]
	states = Vector{Vector{T}}()
	save_history && push!(states, copy(u))

	tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
	t = t0

	if constant_operator
		Aconst, bsteady, Aω1_prev, Aω2_prev, M1, M2 = _prepare_constant_unsteady_two_phase(model, t0, dt, θ)
		sys = LinearSystem(Aconst, copy(bsteady); x=copy(u))
		while t < tend - tol
			dt_step = min(dt, tend - t)
			if abs(dt_step - dt) <= tol
				_set_constant_rhs_two_phase!(sys.b, bsteady, Aω1_prev, Aω2_prev, M1, M2, lay, u, θ)
				solve!(sys; method=method, reuse_factorization=true, kwargs...)
			else
				assemble_unsteady_two_phase!(sys, model, u, t, dt_step, θ)
				solve!(sys; method=method, reuse_factorization=false, kwargs...)
			end
			u .= sys.x
			t += dt_step
			push!(times, t)
			save_history && push!(states, copy(u))
		end
		if !save_history
			states = [copy(u)]
			times = T[t]
		end
		return (times=times, states=states, system=sys, reused_constant_operator=true)
	end

	sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
	while t < tend - tol
		dt_step = min(dt, tend - t)
		assemble_unsteady_two_phase!(sys, model, u, t, dt_step, θ)
		solve!(sys; method=method, reuse_factorization=false, kwargs...)
		u .= sys.x
		t += dt_step
		push!(times, t)
		save_history && push!(states, copy(u))
	end
	if !save_history
		states = [copy(u)]
		times = T[t]
	end
	return (times=times, states=states, system=sys, reused_constant_operator=false)
end

"""
    solve_unsteady_moving!(model, u0, tspan; dt, scheme=:BE, method=:direct, save_history=true, kwargs...)

Time-integrate moving mono or moving two-phase transport.
Accepted `scheme` values are `:BE`, `:CN`, or numeric `theta` in `[0,1]`.
Returns `(times, states, system, reused_constant_operator=false)`.
"""
function solve_unsteady_moving!(
	model::MovingTransportModelMono{N,T},
	u0,
	tspan::Tuple{T,T};
	dt::T,
	scheme=:BE,
	method::Symbol=:direct,
	save_history::Bool=true,
	kwargs...,
) where {N,T}
	t0, tend = tspan
	tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
	dt > zero(T) || throw(ArgumentError("dt must be positive"))
	_resolve_theta(scheme) # validate once at entry

	u = _init_unsteady_state_moving(model, u0)
	lay = model.layout.offsets
	nsys = maximum((last(lay.ω), last(lay.γ)))

	times = T[t0]
	states = Vector{Vector{T}}()
	save_history && push!(states, copy(u))

	sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
	tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
	t = t0
	while t < tend - tol
		dt_step = min(dt, tend - t)
		assemble_unsteady_mono_moving!(sys, model, u, t, dt_step, scheme)
		solve!(sys; method=method, reuse_factorization=false, kwargs...)
		u .= sys.x
		t += dt_step
		push!(times, t)
		save_history && push!(states, copy(u))
	end
	if !save_history
		states = [copy(u)]
		times = T[t]
	end
	return (times=times, states=states, system=sys, reused_constant_operator=false)
end

function solve_unsteady_moving!(
	model::MovingTransportModelTwoPhase{N,T},
	u0,
	tspan::Tuple{T,T};
	dt::T,
	scheme=:BE,
	method::Symbol=:direct,
	save_history::Bool=true,
	kwargs...,
) where {N,T}
	t0, tend = tspan
	tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
	dt > zero(T) || throw(ArgumentError("dt must be positive"))
	_resolve_theta(scheme) # validate once at entry

	u = _init_unsteady_state_moving(model, u0)
	lay = model.layout
	nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

	times = T[t0]
	states = Vector{Vector{T}}()
	save_history && push!(states, copy(u))

	sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
	tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
	t = t0
	while t < tend - tol
		dt_step = min(dt, tend - t)
		assemble_unsteady_two_phase_moving!(sys, model, u, t, dt_step, scheme)
		solve!(sys; method=method, reuse_factorization=false, kwargs...)
		u .= sys.x
		t += dt_step
		push!(times, t)
		save_history && push!(states, copy(u))
	end
	if !save_history
		states = [copy(u)]
		times = T[t]
	end
	return (times=times, states=states, system=sys, reused_constant_operator=false)
end

"""
    omega1_view(model, x)

View the phase-1 bulk block (`ω1`) from a full two-phase state `x`.
The expected unknown ordering is `(ω1, γ1, ω2, γ2)`.
"""
omega1_view(model::TransportModelTwoPhase, x) = x[model.layout.ω1]

"""
    gamma1_view(model, x)

View the phase-1 interface block (`γ1`) from a full two-phase state `x`.
The expected unknown ordering is `(ω1, γ1, ω2, γ2)`.
"""
gamma1_view(model::TransportModelTwoPhase, x) = x[model.layout.γ1]

"""
    omega2_view(model, x)

View the phase-2 bulk block (`ω2`) from a full two-phase state `x`.
The expected unknown ordering is `(ω1, γ1, ω2, γ2)`.
"""
omega2_view(model::TransportModelTwoPhase, x) = x[model.layout.ω2]

"""
    gamma2_view(model, x)

View the phase-2 interface block (`γ2`) from a full two-phase state `x`.
The expected unknown ordering is `(ω1, γ1, ω2, γ2)`.
"""
gamma2_view(model::TransportModelTwoPhase, x) = x[model.layout.γ2]

end
