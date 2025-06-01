@testset "Gram-Schmidt" begin

    include("gram-schmidt.jl")   # importe la fonction gs_classique

    # Exemple minimal pour vecteurs linéairement indépendants
    U = [ [1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0],
          [1.0, 1.0, 1.0] ]

    U1 = [ [1.0, 0.0, 0.0],
           [1.0, 0.0, 0.0],    # dépendance
           [0.0, 1.0, 1.0] ]

    V = gramschmidt(U)

    # Vérifier que chaque vecteur de V est unitaire
    @test all(norm.(V) .≈ 1.0)

    # Vérifier que V est orthogonal : pour i ≠ j, dot(V[i], V[j]) ≈ 0
    for i in 1:length(V)
        for j in (i+1):length(V)
        @test abs(dot(V[i], V[j])) ≤ 1e-10
        end
    end

end  # fin du testset

# Test pour l’implémentation robuste
#
@testset "Gram-Schmidt robuste" begin
    include("gram-schmidt-robuste.jl")

    # (a) Cas lin. indép. : V1r doit être orthonormé
    U2 = [ [1.0, 0.0, 0.0],
           [1.0, 1.0, 0.0],
           [1.0, 1.0, 1.0] ]
    V1r = gramschmidt_robuste(U2)
    @test all(norm.(V1r) .≈ 1.0)
    for i in 1:length(V)
        for j in (i+1):length(V)
        @test abs(dot(V[i], V[j])) ≤ 1e-10
        end
    end


    # (b) Cas dépendant linéaire : on s’attend à un `nothing` ou à une exception
    U2 = [ [1.0, 0.0, 0.0],
           [1.0, 0.0, 0.0],    # dépendance
           [0.0, 1.0, 1.0] ]
    @test gramschmidt_robuste(U2) === nothing
end