using Adapt
using CairoMakie
using ExactClosure: NavierStokes as NS
using GLMakie
using JET
using JLD2
using Random
using Printf

setup = NS.getsetup()

# uaid = NS.dnsaid(setup)
uaid = NS.dnsaid_project(setup)

# let
#     ucpu = adapt(NS.KernelAbstractions.CPU(), uaid)
#     outdir = "output/navierstokes" |> mkdir
#     save_object("$(outdir)/uaid_project.jld2", ucpu)
# end

# uaid = load_object("output/navierstokes/uaid_project.jld2") |> adapt(NS.defaultbackend());

NS.compute_errors(uaid) |> pairs

NS.plot_spectra(setup, uaid)

NS.plot_errors(setup, uaid)
