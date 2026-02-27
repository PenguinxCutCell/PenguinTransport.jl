@testset "Active-cell masking" begin
    moments = build_cut_moments()
    prob = PenguinTransport.TransportProblem(;
        kappa=0.0,
        vel_omega=(0.0, 0.0),
        vel_gamma=(0.0, 0.0),
    )
    sys = PenguinTransport.build_system(moments, prob)

    dims = ntuple(d -> length(moments.xyz[d]), 2)
    V = Float64.(moments.V)
    vtol = sqrt(eps(Float64)) * maximum(abs, V; init=0.0)
    material_mask = (moments.cell_type .!= 0) .& (V .> vtol)
    expected = findall(material_mask .& .!PenguinTransport.padded_mask(dims))

    @test sys.dof_omega.indices == expected
    @test all(diag(sys.M) .> 0.0)
end
