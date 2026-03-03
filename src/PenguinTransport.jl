module PenguinTransport

using LinearAlgebra
using SparseArrays
using StaticArrays

using CartesianGeometry
using CartesianOperators
using PenguinBCs
using PenguinSolverCore

export TransportModelMono
export assemble_steady_mono!, assemble_unsteady_mono!
export solve_steady!, solve_unsteady!
export update_advection_ops!, rebuild!

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
	ops = advection_ops(cap, uωv, uγv; periodic=periodic, scheme=scheme)
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

function _source_values_mono(cap::AssembledCapacity{N,T}, source, t::T) where {N,T}
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

_interface_time_dependent(bc_interface, x::SVector{N,T}) where {N,T} = _value_time_dependent(bc_interface, x)
_interface_time_dependent(::Nothing, x::SVector{N,T}) where {N,T} = false

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

function _mono_row_activity(
	cap::AssembledCapacity{N,T},
	lay,
) where {N,T}
	activeω, activeγ = _cell_activity_masks(cap)
	nsys = maximum((last(lay.ω), last(lay.γ)))
	active = falses(nsys)
	@inbounds for i in 1:cap.ntotal
		active[lay.ω[i]] = activeω[i]
		active[lay.γ[i]] = activeγ[i]
	end
	return active
end

function _validate_mono_layout(cap::AssembledCapacity, lay)
	nt = cap.ntotal
	length(lay.ω) == nt || throw(ArgumentError("layout ω length ($(length(lay.ω))) must match cap.ntotal ($nt)"))
	length(lay.γ) == nt || throw(ArgumentError("layout γ length ($(length(lay.γ))) must match cap.ntotal ($nt)"))
	return nothing
end

function _interface_closure(
	model::TransportModelMono{N,T},
	uγ::NTuple{N,Vector{T}},
	t::T,
) where {N,T}
	nt = model.cap.ntotal
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

		s = zero(T)
		@inbounds for d in 1:N
			s += uγ[d][lin] * model.cap.n_γ[lin][d]
		end

		if s < zero(T)
			g = model.bc_interface === nothing ? zero(T) : _eval_fun_or_const(model.bc_interface, model.cap.C_γ[lin], t)
			a22[lin] = Γ
			b2[lin] = Γ * g
		else
			a21[lin] = -Γ
			a22[lin] = Γ
		end
	end

	return spdiagm(0 => a21), spdiagm(0 => a22), b2
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
	layout::UnknownLayout=layout_mono(cap.ntotal),
) where {N,T}
	validate_borderconditions!(bc_border, N)
	length(b) >= last(layout.offsets.ω) || throw(ArgumentError("rhs vector does not contain ω block"))

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
				row = layout.offsets.ω[row_lin]
				Aface = cap.buf.A[d][row_lin]
				if !(isfinite(Aface) && Aface != zero(T))
					continue
				end

				un = convert(T, normal_sign) * uω[d][row_lin]
				un < zero(T) || continue

				xface = _normal_face_point(cap, I, d, is_high)
				g = convert(T, eval_bc(side_bc.value, xface, t))

				if is_high
					Ihalo = CartesianIndex(ntuple(k -> k == d ? cap.nnodes[d] : I[k], N))
					halo_lin = LI[Ihalo]
					halo_row = layout.offsets.ω[halo_lin]
					_set_row_identity!(A, b, halo_row, g)
				else
					if adv_scheme isa Centered
						b[row] += (un * Aface) * (g / convert(T, 2))
					elseif adv_scheme isa Upwind1
						b[row] += (un * Aface) * g
					else
						throw(ArgumentError("unknown advection scheme $(typeof(adv_scheme))"))
					end
				end
			end
		end
	end

	return A, b
end

function _is_canonical_mono_layout(lay, nt::Int)
	return lay.ω == (1:nt) && lay.γ == ((nt + 1):(2 * nt))
end

function _ops_for_time(model::TransportModelMono{N,T}, t::T) where {N,T}
	uωv, uγv = _velocity_values(model.cap, model.uω, model.uγ, t)
	ops = advection_ops(model.cap, uωv, uγv; periodic=model.periodic, scheme=model.scheme)
	return ops, uωv, uγv
end

function update_advection_ops!(model::TransportModelMono{N,T}; t::T=zero(T)) where {N,T}
	ops, _, _ = _ops_for_time(model, t)
	model.ops = ops
	return model
end

function rebuild!(model::TransportModelMono{N,T}, moments; bc=zero(T), t::T=zero(T)) where {N,T}
	CartesianOperators.rebuild!(model.cap, moments; bc=bc)
	update_advection_ops!(model; t=t)
	return model
end

function assemble_steady_mono!(sys::LinearSystem{T}, model::TransportModelMono{N,T}, t::T) where {N,T}
	nt = model.cap.ntotal
	lay = model.layout.offsets
	_validate_mono_layout(model.cap, lay)
	nsys = maximum((last(lay.ω), last(lay.γ)))

	ops, uωv, uγv = _ops_for_time(model, t)
	model.ops = ops

	conv_bulk = reduce(+, ops.C)
	conv_iface = convert(T, 0.5) * reduce(+, ops.K)

	fω = _source_values_mono(model.cap, model.source, t)
	b1 = model.cap.V * fω
	A21, A22, b2 = _interface_closure(model, uγv, t)
	A11 = conv_bulk + conv_iface
	A12 = conv_iface

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

	apply_box_bc_transport_mono!(sys.A, sys.b, model.cap, uωv, model.bc_border, model.scheme; t=t, layout=model.layout)
	active_rows = _mono_row_activity(model.cap, lay)
	sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
	return sys
end

function _theta_from_scheme(::Type{T}, scheme) where {T}
	if scheme isa Symbol
		if scheme === :BE
			return one(T)
		elseif scheme === :CN
			return convert(T, 0.5)
		end
		throw(ArgumentError("unknown scheme `$scheme`; expected :BE or :CN"))
	elseif scheme isa Real
		return convert(T, scheme)
	end
	throw(ArgumentError("scheme must be a Symbol (:BE/:CN) or a numeric theta"))
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

function assemble_unsteady_mono!(
	sys::LinearSystem{T},
	model::TransportModelMono{N,T},
	uⁿ,
	t::T,
	dt::T,
	scheme_or_theta,
) where {N,T}
	θ = _theta_from_scheme(T, scheme_or_theta)
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

function PenguinSolverCore.assemble!(sys::LinearSystem{T}, model::TransportModelMono{N,T}, t, dt) where {N,T}
	assemble_unsteady_mono!(sys, model, sys.x, convert(T, t), convert(T, dt), one(T))
end

function solve_steady!(model::TransportModelMono{N,T}; t::T=zero(T), method::Symbol=:direct, kwargs...) where {N,T}
	n = maximum((last(model.layout.offsets.ω), last(model.layout.offsets.γ)))
	sys = LinearSystem(spzeros(T, n, n), zeros(T, n))
	assemble_steady_mono!(sys, model, t)
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
	θ = _theta_from_scheme(T, scheme)

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

end
