function PenguinSolverCore.rebuild!(sys::TransportSystem{N,T}, u, p, t) where {N,T}
    if sys.ops_dirty
        sys.ops_diff = CartesianOperators.kernel_ops(sys.moments; bc=sys.bc_diff)
        sys.bc_diff = sys.ops_diff.bc

        ops_adv, bc_adv = _build_ops_adv(sys.moments, sys.bc_adv)
        sys.ops_adv = ops_adv
        sys.bc_adv = bc_adv
        sys.adv_periodic = _adv_periodic_pattern(bc_adv)

        sys.work_diff = CartesianOperators.KernelWork(sys.ops_diff)
        sys.work_adv = CartesianOperators.KernelWork(sys.ops_adv)
        sys.ops_dirty = false
    end

    sys.rebuild_calls += 1
    return nothing
end
