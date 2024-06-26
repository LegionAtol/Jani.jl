export ChemicalSymbolsToInts

struct ChemicalSymbolsToInts
    rev_species::OrderedDict{String, Int}

    function ChemicalSymbolsToInts(all_species::Vector{String})
        rev_species = OrderedDict{String, Int}()
        for (i, species) in enumerate(all_species)
            rev_species[species] = i - 1  # -1 is to start from 0 like in py
        end
        new(rev_species)
    end
end

struct EnergyShifter
    self_energies::AbstractArray{Float64, 1}
    fit_intercept::Bool

    function EnergyShifter(self_energies::Array{Float64, 1}; fit_intercept=false)
        new(gpu(self_energies), fit_intercept)
    end
end

#=  Is a call method instead of forward() better?
function (csi::ChemicalSymbolsToInts)(species::Vector{String})
    rev = [csi.rev_species[s] for s in species]
    return convert(Vector{Int}, rev)
end

function forward(csi::ChemicalSymbolsToInts, species::Vector{String})
    rev = [csi.rev_species[s] for s in species]
    return convert(Vector{Int}, rev)
end
=#

Base.length(csi::ChemicalSymbolsToInts) = length(csi.rev_species)

const PERIODIC_TABLE = (split(strip("""
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """),))