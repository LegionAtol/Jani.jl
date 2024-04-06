export AEVComputer

using LinearAlgebra
using CUDA
using Flux

function compute_shifts(cell::Array, pbc::Array{Bool}, cutoff::Float64)
    reciprocal_cell = inv(transpose(cell))
    inv_distances = vec(norm.(eachrow(reciprocal_cell), 2))
    num_repeats = ceil.(Int, cutoff .* inv_distances)
    num_repeats = ifelse.(pbc, num_repeats, zeros(Int, length(num_repeats)))
    r1 = 1:num_repeats[1]
    r2 = 1:num_repeats[2]
    r3 = 1:num_repeats[3]
    o = [0]
 
    # TODO: maybe to be refactored
    prod1 = [(x, y, z) for x in r1, y in r2, z in r3]
    prod2 = [(x, y, z) for x in r1, y in r2, z in o]
    prod3 = [(x, y, z) for x in r1, y in r2, z in -r3]
    prod4 = [(x, y, z) for x in r1, y in o, z in r3]
    prod5 = [(x, y, z) for x in r1, y in o, z in o]
    prod6 = [(x, y, z) for x in r1, y in o, z in -r3]
    prod7 = [(x, y, z) for x in r1, y in -r2, z in r3]
    prod8 = [(x, y, z) for x in r1, y in -r2, z in o]
    prod9 = [(x, y, z) for x in r1, y in -r2, z in -r3]
    prod10 = [(x, y, z) for x in o, y in r2, z in r3]
    prod11 = [(x, y, z) for x in o, y in r2, z in o]
    prod12 = [(x, y, z) for x in o, y in r2, z in -r3]
    prod13 = [(x, y, z) for x in o, y in o, z in r3]

    prod_vcat1 = vcat(prod1...)
    prod_vcat2 = vcat(prod2...)
    prod_vcat3 = vcat(prod3...)
    prod_vcat4 = vcat(prod4...)
    prod_vcat5 = vcat(prod5...)
    prod_vcat6 = vcat(prod6...)
    prod_vcat7 = vcat(prod7...)
    prod_vcat8 = vcat(prod8...)
    prod_vcat9 = vcat(prod9...)
    prod_vcat10 = vcat(prod10...)
    prod_vcat11 = vcat(prod11...)
    prod_vcat12 = vcat(prod12...)
    prod_vcat13 = vcat(prod13...)

    shifts = vcat(prod_vcat1,prod_vcat2,prod_vcat3,prod_vcat4,prod_vcat5,prod_vcat6,prod_vcat7,prod_vcat8,prod_vcat9,prod_vcat10,prod_vcat11,prod_vcat12,prod_vcat13)
    return shifts
end

function compute_triu_index(num_species::Int)
    ret = zeros(Int64, num_species, num_species)
    count = 0
    for i in 1:num_species, j in i:num_species
        ret[i, j] = count
        count += 1
    end
    ret += transpose(triu(ret, 1))
    return ret
end

mutable struct AEVComputer
    Rcr::Float64
    Rca::Float64
    num_species::Int
    use_cuda_extension::Bool
    EtaR::AbstractArray
    ShfR::AbstractArray
    EtaA::AbstractArray
    Zeta::AbstractArray
    ShfA::AbstractArray
    ShfZ::AbstractArray
    radial_sublength::Int
    radial_length::Int
    angular_sublength::Int
    angular_length::Int
    aev_length::Int
    sizes::Tuple{Int,Int,Int,Int,Int}
    triu_index
    default_cell
    default_shifts
    cuaev_computer::Union{Any, Nothing}
    cuaev_enabled::Bool

    function AEVComputer(;Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=false)
        @assert Rca <= Rcr "Current implementation of AEVComputer assumes Rca <= Rcr"
        
        has_cuaev = false   #TODO: see has_cuaev in python
        # cuda aev
        if use_cuda_extension
            # Assicurati che has_cuaev sia definito prima di questo punto, forse come una variabile globale o come risultato di una funzione
            @assert has_cuaev "Warning: AEV CUDA is not installed. Falling back to CPU."
        else
            println("CUDA extension not used or cuaev is not available.")
        end

        # Convert constant arrays to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        EtaR_conv = gpu(reshape(EtaR, :, 1))
        ShfR_conv = gpu(reshape(ShfR, 1, :))
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        EtaA_conv = gpu(reshape(EtaA, :, 1, 1, 1))
        Zeta_conv = gpu(reshape(Zeta, 1, :, 1, 1))
        ShfA_conv = gpu(reshape(ShfA, 1, 1, :, 1))
        ShfZ_conv = gpu(reshape(ShfZ, 1, 1, 1, :))
        
        # The length of radial subaev of a single species
        radial_sublength = length(EtaR_conv) * length(ShfR_conv)
        # The length of full radial aev
        radial_length = num_species * radial_sublength
        # The length of angular subaev of a single species
        angular_sublength = length(EtaA_conv) * length(Zeta_conv) * length(ShfA_conv) * length(ShfZ_conv)
        # The length of full angular aev
        angular_length = div(num_species * (num_species + 1), 2) * angular_sublength
        # The length of full aev
        aev_length = radial_length + angular_length
        sizes = (num_species, radial_sublength, radial_length, angular_sublength, angular_length)

        triu_index = gpu(compute_triu_index(num_species))

        # Set up default cell and compute default shifts.
        # These values are used when cell and pbc switch are not given.
        cutoff = max(Rcr, Rca)
        default_cell = gpu(Matrix{Float64}(I, 3, 3))
        default_pbc = gpu(zeros(Bool, 3))
        default_shifts = gpu(compute_shifts(default_cell, default_pbc, cutoff))

        if has_cuaev
            cuaev_computer= init_cuaev_computer()
        else
            cuaev_computer = nothing
        end
        cuaev_enabled = use_cuda_extension ? true : false
        
        new(Rcr, Rca, num_species, use_cuda_extension, EtaR_conv, ShfR_conv, EtaA_conv, Zeta_conv, ShfA_conv, ShfZ_conv, radial_sublength, radial_length, angular_sublength, angular_length, aev_length, sizes, triu_index, default_cell, default_shifts, cuaev_computer, cuaev_enabled)

    end
end


function init_cuaev_computer()
    # TODO: CUDA aev
    # return cuaev.CuaevComputer(self.Rcr, self.Rca, self.EtaR.flatten(), self.ShfR.flatten(), self.EtaA.flatten(), self.Zeta.flatten(), self.ShfA.flatten(), self.ShfZ.flatten(), self.num_species)
end