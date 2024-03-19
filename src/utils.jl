export ChemicalSymbolsToInts

struct ChemicalSymbolsToInts
    rev_species::Dict{String, Int}
    #TODO: _dummy::Vector{Float64} or ::CuArray{Float32, 0} useful?

    function ChemicalSymbolsToInts(all_species::Vector{String})
        rev_species = Dict(species => i-1 for (i, species) in enumerate(all_species)) # -1 is to start from 0 like in py
        new(rev_species)
    end
end

function forward(csi::ChemicalSymbolsToInts, species::Vector{String})
    rev = [csi.rev_species[s] for s in species]
    return convert(Vector{Int}, rev)
end

#=  Is a call method instead of forward() better?
function (csi::ChemicalSymbolsToInts)(species::Vector{String})
    rev = [csi.rev_species[s] for s in species]
    return convert(Vector{Int}, rev)
end
=#

Base.length(csi::ChemicalSymbolsToInts) = length(csi.rev_species)