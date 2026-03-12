using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinTransport

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (81, 81))
nt = prod(grid.n)

g = 2.0
body(x, y, t) = sqrt((x - (0.5 + 0.05 * sin(2pi * t)))^2 + (y - 0.5)^2) - 0.22

uω = (zeros(nt), zeros(nt))
uγ = (ones(nt), zeros(nt))
wγ = (zeros(nt), zeros(nt))

model = MovingTransportModelMono(
    grid, body, uω, uγ;
    wγ=wγ,
    source=0.0,
    bc_border=BorderConditions(; left=Outflow(), right=Outflow(), bottom=Outflow(), top=Outflow()),
    bc_interface=Inflow(g),
    scheme=Centered(),
)

u0 = zeros(nt)
dt = 0.02
sys = solve_unsteady_moving!(model, u0, (0.0, dt); dt=dt, scheme=:BE, save_history=false).system
cap = model.cap_slab
@assert !(cap === nothing)
lay = model.layout.offsets

nin = 0
nout = 0
for i in 1:nt
    Γ = cap.buf.Γ[i]
    (isfinite(Γ) && Γ > 0) || continue
    λ = (uγ[1][i] - wγ[1][i]) * cap.n_γ[i][1] + (uγ[2][i] - wγ[2][i]) * cap.n_γ[i][2]
    if λ < 0
        nin += 1
    else
        nout += 1
    end
end

γvals = sys.x[lay.γ]
println("moving mono interface inflow demo")
println("  inflow interface cells  = ", nin)
println("  outflow interface cells = ", nout)
println("  mean(γ)                = ", sum(γvals) / length(γvals))
println("  note: inflow/outflow classification uses λ = (uγ - wγ)·nγ")
