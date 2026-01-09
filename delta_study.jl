# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

using Adapt
using CairoMakie
using ExactClosure: Burgers as B
using JLD2
using LinearAlgebra
using Random
# using WGLMakie
using GLMakie

# outdir = "~/Projects/StructuralErrorPaper" |> expanduser
outdir = joinpath(@__DIR__, "output", "Burgers") |> mkpath
plotdir = "$outdir/figures" |> mkpath

setup = let
    s = B.getsetup()
    outdir = joinpath(s.outdir, "delta-study") |> mkpath
    Δ_ratios = [0, 1, 2, 4, 8, 16, 32, 64]
    nsample = 5 # 0
    (; s..., outdir, Δ_ratios, nsample)
end
setup |> pairs

# DNS-aided LES
series =
    map([:tophat, :gaussian]) do lesfiltertype
        outdir = joinpath(setup.outdir, "delta-study-$lesfiltertype") |> mkpath
        s = (; setup..., outdir)
        B.run_dns_aided_les(s)
        dnsaid = B.load_dns_aided_les(s)
        lesfiltertype => dnsaid
    end |> NamedTuple;

series.tophat |> size
series.tophat[1]

# save_object("$(setup.outdir)/delta-series.jld2", series)
# series = load_object("$(setup.outdir)/delta-series.jld2")

# Compute relative errors
errseries = map(series) do series
    fields = map([:nomodel, :class_m, :class_p, :swapfil]) do key
        e = map(series) do (; fields)
            (; u) = fields[key]
            uref = fields.dns_fil.u
            norm(u - uref) / norm(uref)
        end
        key => (; e, series[1].fields[key].label)
    end
    (; nH = setup.nH, Δ_ratios = setup.Δ_ratios, fields = (; fields...))
end;

let
    for (key, errseries) in zip(keys(errseries), errseries)
        fig = Figure(; size = (950, 500))
        for (i, nH) in enumerate(errseries.nH)
            ax_lin = Axis(
                fig[1, i];
                # yscale = log10,
                xticklabelsvisible = false,
                xticksvisible = false,
                ylabel = "Relative error",
                ylabelvisible = i == 1,
                # yticklabelsvisible = i == 1,
                title = "N = $nH",
                xticks = (eachindex(setup.Δ_ratios), string.(setup.Δ_ratios)),
            )
            ax_log = Axis(
                fig[2, i];
                yscale = log10,
                xlabel = "Δ / h",
                ylabel = "Relative error",
                ylabelvisible = i == 1,
                # yticklabelsvisible = i == 1,
                xticks = (eachindex(setup.Δ_ratios), string.(setup.Δ_ratios)),
            )
            # ylims!(ax_lin, -0.01, 0.12)
            ylims!(ax_log, 1e-16, 1e1)
            for ax in (ax_lin, ax_log)
                vspan!(
                    ax,
                    1,
                    4;
                    alpha = 0.3,
                    color = Cycled(5),
                    label = "Common filter widths",
                )
            end
            widths = setup.Δ_ratios
            for (j, key) in [:nomodel, :class_m, :class_p, :swapfil] |> enumerate
                e = errseries.fields[key].e[i, :]
                color = Cycled(j)
                label = errseries.fields[key].label
                key != :nomodel && scatterlines!(ax_lin, e; label, color)
                # scatterlines!(ax_lin, e; label, color)
                scatterlines!(ax_log, e; label, color)
            end
            # ii = 2:8
            # lines!(ax, ii, map(x -> 3e-2 * 2.0^(-1(x-1)), ii))
            i == 1 && Legend(
                fig[0, 1:3],
                ax_log;
                tellwidth = false,
                orientation = :horizontal,
                framevisible = false,
            )
        end
        rowgap!(fig.layout, 10)
        save("$(setup.plotdir)/burgers_delta_errors_$(key).pdf", fig; backend = CairoMakie)
        display(fig)
        fig
    end
end
