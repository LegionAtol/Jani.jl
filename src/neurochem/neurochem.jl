module NeuroChem
    #include("parse_resources.jl")
    using parse_resources
    
    export parse_neurochem_resources, Constants, load_sae, load_model

    struct Constants
    end

    function load_sae(filename::String, return_dict::Bool=false)
        #=
        if return_dict
            return EnergyShifter(self_energies), d
        end
        return EnergyShifter(self_energies)
        =#
    end
    
    function load_model(species::Vector{String}, dir_::String)
        #return ANIModel(models)
    end
end