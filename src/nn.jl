export SpeciesConverter, SpeciesEnergies

mutable struct SpeciesEnergies
    species::Array
    energies::Array
end

struct SpeciesConverter
    conv_tensor::Vector{Int}

    function SpeciesConverter(species::Vector{String})
        rev_idx = Dict(s => k for (k, s) in enumerate(PERIODIC_TABLE))
        maxidx = maximum(values(rev_idx))
        conv_tensor = fill(-1, maxidx + 2)
        for (i, s) in enumerate(species)
            conv_tensor[rev_idx[s]] = i-1 #the first element H will be 0
        end
        new(conv_tensor)
    end
end