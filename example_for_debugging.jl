#=
This file serves only as example code.
It can be used to start debugging the code from here
=#

using CUDA
import Jani

device = has_cuda() ? CuDevice(0) : nothing

model = Jani.ANI2x(true) |> to(device)