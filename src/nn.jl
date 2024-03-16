export SpeciesConverter, SpeciesEnergies

mutable struct SpeciesEnergies
    species::Array
    energies::Array
end

mutable struct SpeciesConverter
    conv_tensor::Array
end