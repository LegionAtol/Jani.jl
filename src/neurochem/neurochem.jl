export parse_neurochem_resources, Constants, load_sae, load_model
include("../utils.jl") #TODO: without this include(), ChemicalSymbolsToInts is not seen 

mutable struct Constants
    filename::String
    Rcr::Float64
    Rca::Float64
    EtaR::Array{Float64}
    ShfR::Array{Float64}
    EtaA::Array{Float64}
    Zeta::Array{Float64}
    ShfA::Array{Float64}
    ShfZ::Array{Float64}
    species::Vector{String}
    num_species::Int
    species_to_tensor::ChemicalSymbolsToInts

    function Constants(filename::String)
        c = new(filename)
        f = open(filename)
        for i in eachline(f)
            try
                line = split(i, '=')
                name = strip(line[1])
                value = strip(line[2])
                if name in ["Rcr", "Rca"]
                    setfield!(c, Symbol(name), parse(Float64, value))
                elseif name in ["EtaR", "ShfR", "EtaA", "Zeta", "ShfA", "ShfZ"]
                    value = replace(replace(value, '[' => ""), ']' => "")
                    # Split string only if it contains a comma
                    if contains(value, ',')
                        value = [parse(Float64, strip(String(x))) for x in split(value, ',')]
                    else
                        value = [parse(Float64, value)]
                    end
                    variable_map = Dict(
                        "EtaR" => :EtaR,
                        "ShfR" => :ShfR,
                        "Zeta" => :Zeta,
                        "ShfZ" => :ShfZ,
                        "EtaA" => :EtaA,
                        "ShfA" => :ShfA
                    )
                    if haskey(variable_map, name)
                        setfield!(c, variable_map[name], value)
                    end
                elseif name == "Atyp"
                    value = [replace(x, r"[\[\]]" => "") for x in split(value, ',')]
                    c.species = value
                end
            catch e
                throw(ArgumentError("unable to parse const file: $e"))
            end
        end
        close(f)

        c.num_species = length(c.species)
        c.species_to_tensor = ChemicalSymbolsToInts(c.species)
        return c
    end
end

Base.length(c::Constants) = return 9

Base.keys(c::Constants) = [:Rcr, :Rca, :EtaR, :ShfR, :EtaA, :Zeta, :ShfA, :ShfZ, :num_species]

Base.values(c::Constants) = [getfield(c, key) for key in keys(c)]

Base.pairs(c::Constants) = [key => getfield(c, key) for key in keys(c)]

function load_sae(filename::String; return_dict::Bool=false)
    self_energies = []
    d = OrderedDict{String, Float64}()

    open(filename) do f
        for line in eachline(f)
            parts = strip.(split(line, '='))
            species, index_str = strip.(split(parts[1], ','))
            index = parse(Int, index_str)
            value = parse(Float64, parts[2])
            d[species] = value
            push!(self_energies, (index, value))
        end
    end

    sorted_energies = map(x -> x[2], sort(self_energies, by=x->x[1]))

    if return_dict
        return EnergyShifter(sorted_energies), d
    else
        return EnergyShifter(sorted_energies)
    end
end

function load_model_ensemble(species::Vector{String}, prefix::String, count::Int)
    models = []
    for i in 0:(count - 1)  # start from 0
        network_dir = joinpath(prefix * string(i), "networks")
    #    push!(models, load_model(species, network_dir))
    end
    #return Ensemble(models)
end

function load_model(species::Vector{String}, dir_::String)
    #return ANIModel(models)
end