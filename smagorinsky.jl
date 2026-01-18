# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

using CairoMakie
using ExactClosure
using ExactClosure: Burgers as B
using FFTW
using JLD2
using KernelDensity
using LinearAlgebra
using Random
# using WGLMakie
using GLMakie

setup = (; B.getsetup()..., Δ_ratios = [0, 2])
setup |> pairs

dnsdata = B.create_dns(setup; cfl_factor = 0.3);

# save_object("$(setup.outdir)/burgers-dns.jld2", dnsdata)
# dnsdata = load_object("$(setup.outdir)/burgers-dns.jld2")

fields = B.smagorinsky_fields(setup, dnsdata);

# Estimate θ
θ_classic = map(fields) do (; Sh, Th, D_Sh, D_Th)
    θ2 = -dot(Sh, Th) / dot(Sh, Sh)
    # θ2 = -dot(D_Sh, D_Th) / dot(D_Sh, D_Sh)
    # θ2 = -sum(D_Th) / sum(D_Sh)
    sqrt(θ2)
end

θ_discret = map(fields) do (; SH, TH, D_SH, D_TH)
    θ2 = -dot(SH, TH) / dot(SH, SH)
    # θ2 = -dot(D_SH, D_TH) / dot(D_SH, D_SH)
    # θ2 = -sum(D_TH) / sum(D_SH)
    sqrt(θ2)
end

u_classic = B.solve_smagorinsky(setup, dnsdata, θ_classic)
u_discret = B.solve_smagorinsky(setup, dnsdata, θ_discret)

# save_object("$(setup.outdir)/burgers_classic.jld2", u_classic)
# save_object("$(setup.outdir)/burgers_discret.jld2", u_discret)

# u_classic = load_object("$(setup.outdir)/burgers_classic.jld2")
# u_discret = load_object("$(setup.outdir)/burgers_discret.jld2")

# Compute relative errors
relerrs = map(Iterators.product(eachindex(setup.Δ_ratios), eachindex(setup.nH))) do (iΔ, iH)
    Δ_ratio = setup.Δ_ratios[iΔ]
    nH = setup.nH[iH]
    gh = B.Grid(setup.L, setup.nh)
    gH = B.Grid(setup.L, nH)
    @show Δ_ratio nH
    U = dnsdata[2]
    comp = div(gh.n, gH.n)
    Ubar = fields[iΔ, iH].Ubar[comp:comp:end, :] # Extract coarse-grid components
    U_classic = u_classic[iΔ, iH]
    U_discret = u_discret[iΔ, iH]
    e = map((; classic = U_classic, discret = U_discret)) do U
        norm(U - Ubar) / norm(Ubar)
    end
    (; nH, e)
end

B.plot_smagorinsky_coefficients(θ_classic, θ_discret, setup)

B.plot_smagorinsky_errors(relerrs, setup)

# Compute spectra
specseries = B.compute_smagorinsky_spectra(fields, u_classic, u_discret, dnsdata, setup)

B.plot_smagorinsky_spectra(specseries, setup)
