using LinearAlgebra

function gramschmidt_robuste(us::Vector{Vector{Float64}}; tol::Float64 = 1e-10)
    # us : vecteur de vecteurs d’entrée (Vector{Vector{Float64}})
    # tol : tolérance pour considérer qu’un résidu est numériquement nul

    n = length(us)
    # On prépare un tableau pour stocker les vecteurs orthonormaux
    vs = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        # 1) On part du i-ème vecteur d’entrée
        v = copy(us[i])

        # 2) Modified Gram–Schmidt : on projette le résidu v sur chacun des vs[j]
        for j in 1:i-1
            # Utiliser dot(v, vs[j]) pour projeter le **résidu courant**, pas la version classique dot(us[i], vs[j])
            coeff = dot(v, vs[j]) / dot(vs[j], vs[j])
            v .-= coeff .* vs[j]   # soustraction élément‐par‐élément, in‐place
        end

        # 3) Vérification de la dépendance linéaire
        if norm(v) ≤ tol
            # Si le résidu est (quasi) nul, alors us[i] est dans l'espace engendré par les vecteurs précédents.
            # On retourne `nothing` 
            return nothing
        end

        # 4) Sinon, on normalise le résidu pour obtenir le nouveau vecteur orthonormal
        vs[i] = v / norm(v)
    end

    return vs
end
