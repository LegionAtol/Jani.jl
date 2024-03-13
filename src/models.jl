using .nn: SpeciesConverter, SpeciesEnergies
using  .aev: AEVComputer

mutable struct SpeciesEnergiesQBC
    species::Tensor
    energies::Tensor
    qbcs::Tensor
end

mutable struct BuiltinModel
    species_converter
    aev_computer
    neural_networks
    energy_shifter
    _species_to_tensor
    species
    periodic_table_index
    consts
    sae_dict
    #TODO: add type
end
#TODO default constructor or not?

#@classmethod
function _from_neurochem_resources(info_file_path, periodic_table_index=false, model_index=0)
    #using neurochem.NeuroChem: parse_neurochem_resources, Constants, load_sae, load_model
    import neurochem.NeuroChem
    const_file, sae_file, ensemble_prefix, ensemble_size = NeuroChem.parse_neurochem_resources(info_file_path)

    if model_index >= ensemble_size
        throw(ArgumentError("The ensemble size is only $ensemble_size, model $model_index can't be loaded"))
    end

    consts = NeuroChem.Constants(const_file)
    species_converter = SpeciesConverter(consts.species) # in nn.jl #Is it better to use a Module for nn.jl?
    aev_computer = AEVComputer(consts...)
    energy_shifter, sae_dict = NeuroChem.load_sae(sae_file, return_dict=true)
    species_to_tensor = consts.species_to_tensor

    network_dir = joinpath("$ensemble_prefix$model_index", "networks")
    neural_networks = NeuroChem.load_model(consts.species, network_dir)

    return BuiltinModel(species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index, consts.species)
end

function forward(self::BuiltinModel, species_coordinates::Tuple{Tensor, Tensor}, cell::Union{Tensor, Nothing}=nothing, pbc::Union{Tensor, Nothing}=nothing)::SpeciesEnergies
    if self.periodic_table_index
        species_coordinates = self.species_converter(species_coordinates)
    end

    # check if unknown species are included
    if any(x -> x ≥ self.aev_computer.num_species, species_coordinates[1])
        throw(ArgumentError("Unknown species found in $(species_coordinates[1])"))
    end

    species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
    species_energies = self.neural_networks(species_aevs)
    return self.energy_shifter(species_energies)
end

#@torch.jit.export
function atomic_energies(self::BuiltinModel, species_coordinates::Tuple{Tensor, Tensor},
    cell::Union{Tensor, Nothing}=nothing,
    pbc::Union{Tensor, Nothing}=nothing)::SpeciesEnergies
end

#@torch.jit.export
function _recast_long_buffers(self::BuiltinModel)
end 

function species_to_tensor(self::BuiltinModel; args...; kwargs...)
end

function ase(self::BuiltinModel; kwargs...)
end

mutable struct BuiltinEnsemble <: BuiltinModel
end

#@classmethod
function _from_neurochem_resources(info_file_path, periodic_table_index=false)
end

function Base.getindex(self::BuiltinEnsemble, index::Int)
    #return BuiltinModel(self.species_converter, self.aev_computer,
    #                   self.neural_networks[index], self.energy_shifter,
    #                   self._species_to_tensor, self.consts, self.sae_dict,
    #                   self.periodic_table_index)
end

# @torch.jit.export
function members_energies(self::BuiltinEnsemble, species_coordinates::Tuple{Tensor, Tensor}, cell::Union{Tensor, Nothing}=nothing, pbc::Union{Tensor, Nothing}=nothing)
end

# @torch.jit.export
function energies_qbcs(self::BuiltinEnsemble, species_coordinates::Tuple{Tensor, Tensor},
                        cell::Union{Tensor, Nothing}=nothing,
                        pbc::Union{Tensor, Nothing}=nothing, unbiased::Bool=true)::SpeciesEnergiesQBC
end

function Base.length(self::BuiltinEnsemble)
    """Get the number of networks in the ensemble

    Returns:
        length (::Int): Number of networks in the ensemble
    """
    return length(self.neural_networks)
end

function ANI1x(periodic_table_index::Bool=false, model_index::Union{Int, Nothing}=nothing)
end

function ANI1ccx(periodic_table_index::Bool=false, model_index::Union{Nothing, Int}=nothing)
end

function ANI2x(periodic_table_index::Bool=false, model_index::Union{Int, Nothing}=nothing)
    info_file = "ani-2x_8x.info"
    if isnothing(model_index)
        return _from_neurochem_resources(info_file, periodic_table_index)
    end
    return _from_neurochem_resources(info_file, periodic_table_index, model_index)
end

