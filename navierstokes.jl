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

let
    ucpu = adapt(NS.KernelAbstractions.CPU(), uaid)
    (; outdir, n_dns) = setup
    file = "$(outdir)/uaid-project-ndns=$(n_dns).jld2"
    save_object(file, ucpu)
end

uaid = let
    file = "$(setup.outdir)/uaid-project-ndns=$(setup.n_dns).jld2"
    o = load_object(file)
    o |> adapt(NS.defaultbackend())
end;

NS.compute_errors(uaid) |> pairs

# :u_nomo     => 0.263051
# :u_c        => 0.211564
# :u_cf       => 0.0642457
# :u_cfd      => 5.68942e-15
# :u_cfd_symm => 0.0285891

NS.plot_spectra(setup, uaid)

NS.plot_errors(setup, uaid)
