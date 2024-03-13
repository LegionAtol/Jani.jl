export SpeciesConverter, SpeciesEnergies

mutable struct SpeciesEnergies
    species::Tensor
    energies::Tensor
end

mutable struct SpeciesConverter
    conv_tensor::Tensor
end