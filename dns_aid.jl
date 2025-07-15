# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

@info "Loading packages"
flush(stderr)

using Adapt
using CUDA
using JLD2
using LinearAlgebra
using Printf
using ExactClosure: NavierStokes
using Turbulox

@info "Loading case"
flush(stderr)

# Choose case
case = NavierStokes.smallcase() # Laptop CPU with 16 GB RAM
case = NavierStokes.mediumcase() # GPU with 24 GB RAM
case = NavierStokes.largecase() # GPU with 90 GB RAM (H100)

(; viscosity, outdir, datadir, plotdir, seed, g_dns, g_les) = case
T = typeof(g_dns.L)

let
    cfl = 0.15 |> T
    tstop = 0.1 |> T
    poisson_dns = poissonsolver(g_dns)
    for (j, g_les) in enumerate(g_les),
        (i, ex) in enumerate(["volavg", "project_volavg", "surfavg"])
        # parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) == i + 3 * (j - 1) || continue
        @info "Running experiment: $(ex)"
        flush(stderr)
        compression = div(g_dns.n, g_les.n)
        poisson_les = poissonsolver(g_les)
        # Load initial DNS
        path = joinpath(outdir, "u.jld2")
        data = path |> load_object |> adapt(g_dns.backend)
        ustart = VectorField(g_dns, data)
        if ex == "volavg"
            sols, relerr = NavierStokes.dns_aid_volavg(;
                ustart,
                g_dns,
                g_les,
                poisson_dns,
                poisson_les,
                viscosity,
                compression,
                doproject = false,
                docopy = false, # Overwrite ustart
                tstop,
                cfl,
            )
        elseif ex == "project_volavg"
            sols, relerr = NavierStokes.dns_aid_volavg(;
                ustart,
                g_dns,
                g_les,
                poisson_dns,
                poisson_les,
                viscosity,
                compression,
                doproject = true,
                docopy = false, # Overwrite ustart
                tstop,
                cfl,
            )
        elseif ex == "surfavg"
            sols, relerr = NavierStokes.dns_aid_surface(;
                ustart,
                g_dns,
                g_les,
                poisson_dns,
                poisson_les,
                viscosity,
                compression,
                doproject = true,
                docopy = false, # Overwrite ustart
                tstop,
                cfl,
            )
        end
        # Save errors
        file = joinpath(datadir, "relerr-$(ex)-$(g_les.n).jld2")
        @info "Saving errors to $(file)"
        flush(stderr)
        jldsave(file; relerr)
        # Compute spectra
        diss = ScalarField(g_dns)
        apply!(Turbulox.dissipation!, g_dns, diss, sols.dns_ref, viscosity)
        D = sum(diss.data) / length(diss)
        diss = nothing # free up memory
        specs = map(sols) do u
            stuff = spectral_stuff(u.grid)
            spectrum(u; stuff)
        end
        file = joinpath(datadir, "spectra-$(ex)-$(g_les.n).jld2")
        @info "Saving spectra to $(file)"
        flush(stderr)
        jldsave(file; specs, D)
        sols = nothing # free up memory
    end
end

open(joinpath(outdir, "ns_error.tex"), "w") do io
    println(io, "Filter & \$N_H\$ & No-model & Classic & Swap-sym & Swap \\\\")
    experiments = ["volavg", "project_volavg", "surfavg"]
    for g_les in g_les, ex in experiments
        relerr = load(joinpath(datadir, "relerr-$(ex)-$(g_les.n).jld2"), "relerr")
        r = map(x -> round(x[end]; sigdigits = 3), relerr)
        println(
            io,
            join(
                [
                    Dict("volavg" => "VA ", "project_volavg" => "PVA", "surfavg" => "SA ")[ex],
                    string(g_les.n),
                    r.nomodel,
                    r.classic,
                    r.swapfil_symm,
                    r.swapfil,
                ],
                " & ",
            )...,
            " \\\\",
        )
    end
end
open(readlines, joinpath(outdir, "ns_error.tex")) .|> println;

@info "Done."
flush(stderr)
