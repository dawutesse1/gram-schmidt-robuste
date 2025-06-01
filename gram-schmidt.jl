using LinearAlgebra

function gramschmidt(us)  # us should be a vector of vectors
    vs = copy(us)
    for i = 1 : length(vs)
        v = vs[i]
        for j = 1 : i - 1
           v .-= dot(us[i], vs[j]) .* vs[j]  # here .= indicates in-place update
        end
        v ./= norm(v)
    end
    return vs
end