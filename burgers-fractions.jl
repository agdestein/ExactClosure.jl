# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

using CairoMakie
using CUDA
using ExactClosure
using ExactClosure.Burgers
using FFTW
using JLD2
using KernelDensity
using LinearAlgebra
using Random
using Statistics
# using WGLMakie
using GLMakie

setup = let
    L = 2π
    nh = 100 * 3^3 * 5
    nH = 100 .* 3 .^ (1:3)
    Δ_ratios = [0, 1, 2, 4, 8, 16, 32]
    visc = 5e-4
    kpeak = 10
    initialenergy = 2.0
    tstop = 0.1
    nsample = 10
    lesfiltertype = :gaussian
    backend = CUDA.functional() ? CUDABackend() : Burgers.AK.KernelAbstractions.CPU()
    # backend = Burgers.AK.KernelAbstractions.CPU()
    outdir = joinpath(@__DIR__, "output", "Burgers") |> mkpath
    plotdir = "$outdir/figures" |> mkpath
    (;
        L,
        nh,
        nH,
        Δ_ratios,
        visc,
        kpeak,
        initialenergy,
        tstop,
        nsample,
        lesfiltertype,
        backend,
        outdir,
        plotdir,
    )
end
setup |> pairs

# DNS-aided LES
dns = Burgers.create_dns(setup; cfl_factor = 0.4)

fractions = Burgers.compute_fractions(dns, setup)

Burgers.plot_fractions(fractions, setup)

fractions[:, 3]
