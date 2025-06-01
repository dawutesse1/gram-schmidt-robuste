using Test           
using LinearAlgebra  

# 1) Chargement de la version classique
include("gram-schmidt.jl")

# 2) Chargement du module robuste
include("gram-schmidt-robuste.jl")

# 3) Tests pour la version classique
@testset "Gram-Schmidt" begin
    U = [ [1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0],
          [1.0, 1.0, 1.0] ]
    V = gramschmidt(U)
    @test all(norm.(V) .≈ 1.0)
    for i in 1:length(V)
        for j in (i+1):length(V)
            @test abs(dot(V[i], V[j])) ≤ 1e-10
        end
    end
end

# 4) Tests pour la version robuste
@testset "Gram-Schmidt robuste" begin
    # (a) Cas lin. indép. : on s’attend à un vecteur orthonormal
    U2 = [ [1.0, 0.0, 0.0],
           [1.0, 1.0, 0.0],
           [1.0, 1.0, 1.0] ]
    V1r = gramschmidt_robuste(U2)
    @test V1r !== nothing
    @test all(norm.(V1r) .≈ 1.0)
    for i in 1:length(V1r)
        for j in (i+1):length(V1r)
            @test abs(dot(V1r[i], V1r[j])) ≤ 1e-10
        end
    end

    # (b) Cas dépendant linéaire : on s’attend à nothing
    U3 = [ [1.0, 0.0, 0.0],
           [1.0, 0.0, 0.0],  # dépendance
           [0.0, 1.0, 1.0] ]
    @test gramschmidt_robuste(U3) === nothing
end
