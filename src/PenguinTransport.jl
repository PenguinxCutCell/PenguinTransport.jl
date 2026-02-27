module PenguinTransport

using LinearAlgebra
using SparseArrays

using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

include("types.jl")
include("build.jl")
include("rhs.jl")
include("rebuild.jl")
include("io.jl")
include("updaters.jl")

export TransportProblem
export TransportSystem
export build_system
export full_state
export KappaUpdater, SchemeUpdater, VelocityUpdater, AdvBCUpdater, SourceUpdater

end
