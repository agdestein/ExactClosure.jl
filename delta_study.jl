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

# outdir = "~/Projects/StructuralErrorPaper" |> expanduser
outdir = joinpath(@__DIR__, "output", "Burgers") |> mkpath
plotdir = "$outdir/figures" |> mkpath

setup = let
    L = 2π
    nh = 100 * 3^3 * 5
    nH = 100 .* 3 .^ (1:3)
    Δ_scalers = [0, 1, 2, 4, 8, 16, 32, 64]
    visc = 5e-4
    kpeak = 10
    initialenergy = 2.0
    tstop = 0.1
    nsample = 10
    backend = CUDA.functional() ? CUDABackend() : Burgers.AK.KernelAbstractions.CPU()
    (; L, nh, nH, Δ_scalers, visc, kpeak, initialenergy, tstop, nsample, backend)
end
setup |> pairs

# DNS-aided LES
series =
    map([:tophat, :gaussian]) do lesfiltertype
        lesfiltertype => map(setup.nH) do nH
            (; L, nh, visc, kpeak, initialenergy, tstop, nsample, Δ_scalers, backend) =
                setup
            gh = Grid(L, nh)
            gH = Grid(L, nH)
            widths = Δ_scalers * spacing(gH)
            nΔ = length(widths)
            fields =
                (;
                    dns_ref = (; g = gh, u = zeros(nh, nsample, nΔ), label = "DNS"),
                    dns_fil = (;
                        g = gH,
                        u = zeros(nH, nsample, nΔ),
                        label = "Filtered DNS",
                    ),
                    nomodel = (; g = gH, u = zeros(nH, nsample, nΔ), label = "No model"),
                    class_m = (; g = gH, u = zeros(nH, nsample, nΔ), label = "Classic"),
                    class_p = (; g = gH, u = zeros(nH, nsample, nΔ), label = "Classic+"),
                    swapfil = (; g = gH, u = zeros(nH, nsample, nΔ), label = "Swap (ours)"),
                ) |> adapt(backend)
            for (i, j) in Iterators.product(1:nsample, eachindex(widths)) |> collect
                @info "Filter: $(lesfiltertype), N = $nH, sample = $i, Δ/h = $(Δ_scalers[j])"
                rng = Xoshiro(i)
                ustart = randomfield(rng, gh, kpeak, initialenergy) |> adapt(backend)
                sols = Burgers.dns_aided_les(
                    ustart,
                    gh,
                    gH,
                    visc;
                    Δ = widths[j],
                    tstop,
                    cfl_factor = 0.3,
                    lesfiltertype,
                )
                for key in keys(fields)
                    copyto!(view(fields[key].u, :, i, j), sols[key])
                end
            end
            (; nH, fields) |> adapt(Burgers.AK.KernelAbstractions.CPU())
        end
    end |> NamedTuple;

# save_object("$outdir/DeltaSeries.jld2", series)
# series = load_object("$outdir/DeltaSeries.jld2")

# Compute relative errors
errseries = map(series) do series
    fields = map([:nomodel, :class_m, :class_p, :swapfil]) do key
        e = map(series) do (; nH, fields)
            (; u) = fields[key]
            map(axes(u, 3)) do i
                ules = selectdim(u, 3, i)
                uref = selectdim(fields.dns_fil.u, 3, i)
                norm(ules - uref) / norm(uref)
            end
        end
        key => (; e, series[1].fields[key].label)
    end
    (; nH = getindex.(series, :nH), fields = (; fields...))
end;
errseries.tophat.fields |> pairs
errseries.gaussian.fields |> pairs

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
                xticks = (eachindex(setup.Δ_scalers), string.(setup.Δ_scalers)),
            )
            ax_log = Axis(
                fig[2, i];
                yscale = log10,
                xlabel = "Δ / h",
                ylabel = "Relative error",
                ylabelvisible = i == 1,
                # yticklabelsvisible = i == 1,
                xticks = (eachindex(setup.Δ_scalers), string.(setup.Δ_scalers)),
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
            widths = setup.Δ_scalers
            for (j, key) in [:nomodel, :class_m, :class_p, :swapfil] |> enumerate
                e = errseries.fields[key].e[i]
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
        save("$plotdir/burgers_delta_errors_$(key).pdf", fig; backend = CairoMakie)
        display(fig)
        fig
    end
end
