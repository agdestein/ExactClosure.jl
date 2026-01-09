# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

using CairoMakie
using ExactClosure: Burgers as B
using JLD2
using KernelDensity
using LinearAlgebra
using Random
# using WGLMakie
using GLMakie

# Replace default Δ_ratios
setup = (; B.getsetup()..., Δ_ratios = [0, 1, 2, 4, 8, 16, 32])
setup |> pairs

dns = B.create_dns(setup; cfl_factor = 0.4)

fractions = B.compute_fractions(dns, setup)

B.plot_fractions(fractions, setup)

fractions[:, 3]
