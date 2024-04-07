using OrderedCollections: OrderedDict

mutable struct SpeciesEnergiesQBC
    species::Array
    energies::Array
    qbcs::Array
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

function _from_neurochem_resources(info_file_path, periodic_table_index=false, model_index=0)
end

function forward(self::BuiltinModel, species_coordinates::Tuple{Array, Array}, cell::Union{Array, Nothing}=nothing, pbc::Union{Array, Nothing}=nothing)::SpeciesEnergies
    #=
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
    =#
end

function atomic_energies(self::BuiltinModel, species_coordinates::Tuple{Array, Array},
                            cell::Union{Array, Nothing}=nothing,
                            pbc::Union{Array, Nothing}=nothing)::SpeciesEnergies
    return nothing
end

function _recast_long_buffers(self::BuiltinModel)
end 

# see .py for args and kwargs
#function species_to_tensor(self::BuiltinModel, args::Union{Array,nothing}=nothing, kwargs::Union{Dict{Any,Any},nothing}=nothing)
#end

function ase(self::BuiltinModel; kwargs...)
end

# maybe useless, (BuiltinModel should be abstract)
#mutable struct BuiltinEnsemble <: BuiltinModel
#end

function _from_neurochem_resources(info_file_path, periodic_table_index=false)
    const_file, sae_file, ensemble_prefix, ensemble_size = parse_neurochem_resources(info_file_path)
    
    consts = Constants(const_file)
    species_converter = SpeciesConverter(consts.species)

    kwargs = OrderedDict{Symbol,Any}()
    for (key, value) in pairs(consts)
        kwargs[key] = value
    end
    aev_computer = AEVComputer(;kwargs...)
    energy_shifter, sae_dict = load_sae(sae_file, return_dict=true)
    species_to_tensor = consts.species_to_tensor
    neural_networks = load_model_ensemble(consts.species, ensemble_prefix, ensemble_size)

    #return BuiltinModel(species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, consts, sae_dict, periodic_table_index, consts.species)
end

function Base.getindex(self::BuiltinModel, index::Int)
end

function members_energies(self::BuiltinModel, species_coordinates::Tuple{Array, Array}, cell::Union{Array, Nothing}=nothing, pbc::Union{Array, Nothing}=nothing)
end

function energies_qbcs(self::BuiltinModel, species_coordinates::Tuple{Array, Array},
                        cell::Union{Array, Nothing}=nothing,
                        pbc::Union{Array, Nothing}=nothing, unbiased::Bool=true)::SpeciesEnergiesQBC
end

function Base.length(self::BuiltinModel)
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
    #can be compacted?, Multiple dispatch
    if isnothing(model_index)
        return _from_neurochem_resources(info_file, periodic_table_index)
    end
    return _from_neurochem_resources(info_file, periodic_table_index, model_index)
end
