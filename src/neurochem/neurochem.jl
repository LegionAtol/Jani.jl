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
        self = new(filename)
        f = open(filename)
        for i in eachline(f)
            try
                line = split(i, '=')
                name = strip(line[1])
                value = strip(line[2])
                if name in ["Rcr", "Rca"]
                    setfield!(self, Symbol(name), parse(Float64, value))
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
                        setfield!(self, variable_map[name], value)
                    end
                elseif name == "Atyp"
                    value = [replace(x, r"[\[\]]" => "") for x in split(value, ',')]
                    self.species = value
                end
            catch e
                throw(ArgumentError("unable to parse const file: $e"))
            end
        end
        close(f)

        self.num_species = length(self.species)
        self.species_to_tensor = ChemicalSymbolsToInts(self.species)
        return self
    end
end

#TODO: function __iter__(self),  __getitem__(self, item)
Base.length(self::Constants) = return 8

function load_sae(filename::String, return_dict::Bool=false)
end

function load_model(species::Vector{String}, dir_::String)
    #return ANIModel(models)
end