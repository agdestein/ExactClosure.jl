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
using ExactClosure.Burgers
using FFTW
using JLD2
using KernelDensity
using LinearAlgebra
using Random
using Statistics
using WGLMakie

# outdir = "~/Projects/StructuralErrorPaper" |> expanduser
outdir = joinpath(@__DIR__, "output", "Burgers") |> mkpath
plotdir = "$outdir/figures" |> mkpath

setup = let
    L = 2π
    nh = 2^5 * 3^5
    nH = 2^5 .* 3 .^ (2:4)
    Δ_ratio = 2
    visc = 5e-4
    kp = 10
    A = 2 / kp / 3 / sqrt(π)
    a = sqrt(2 * A)
    tstop = 0.1
    nsample = 100
    (; L, nh, nH, Δ_ratio, visc, kp, a, tstop, nsample)
end
setup |> pairs

dnsdata = create_dns(setup; cfl_factor = 0.3);

# save_object("$outdir/burgers_dns.jld2", dnsdata)
# dnsdata = load_object("$outdir/burgers_dns.jld2")

smagcoeffs = fit_smagcoeffs(setup, dnsdata; lesfiltertype = :gaussian);
getindex.(smagcoeffs, 1)
@. getindex(smagcoeffs, 1) * setup.L^2 / setup.nH^2 * (1 + setup.Δ_ratio^2)

# 3-element Vector{Float64}:
#  1.3078695085917662
#  0.6567297859704491
#  0.38678361976428427

usmag = Burgers.solve_smagorinsky(setup, dnsdata, smagcoeffs; lesfiltertype = :gaussian)

let
    j = 1
    fig = Figure(; size = (400, 800))
    for (i, nH) in enumerate(setup.nH)
        gh = Grid(setup.L, setup.nh)
        gH = Grid(setup.L, nH)
        xh = points_stag(gh)
        xH = points_stag(gH)
        ax = Axis(fig[i, 1]; title = "N = $nH", xlabel = "x", ylabel = "u")
        lines!(ax, xh, dnsdata[2][:, j]; label = "DNS")
        lines!(ax, xH, usmag[i][:, j]; label = "Smagorinsky")
    end
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x")
    for i = 1:3
        θ, S, T = smagcoeffs[i]
        j = 1
        t = T[:, j]
        lines!(ax, t)
    end
    fig
end

let
    j = 1
    θ, S, T = smagcoeffs[j]
    Ustart, U = dnsdata
    i = 1
    ustart = Ustart[:, i]
    u = U[:, i]
    s = S[:, i]
    t = T[:, i]
    g = Grid(setup.L, setup.nh)
    xs = points_stag(g)
    xc = points_coll(g)
    H = setup.L / setup.nH[j]
    Δ = setup.Δ_ratio * H
    fac = Δ^2 + H^2
    C = 15
    fig = Figure()
    axu = Axis(fig[1, 1]; ylabel = "u", xticklabelsvisible = false)
    lines!(axu, xs, ustart; color = Cycled(2), label = "u initial")
    lines!(axu, xs, u; color = Cycled(1), label = "u final")
    axs = Axis(fig[2, 1]; xticklabelsvisible = false, ylabel = "s")
    lines!(axs, xc, s; label = "Smagorinsky")
    axt = Axis(fig[3, 1]; xlabel = "x", ylabel = "t")
    lines!(axt, xc, t; label = "True SFS")
    lines!(axt, xc, -C * s; label = "Scaled Smagorinsky")
    linkxaxes!(axu, axs, axt)
    xlims!(axt, 2.0, 2.4)
    axislegend(axu)
    axislegend(axs)
    axislegend(axt; position = :rb)
    save("$(plotdir)/burgers_smagorinsky_fit_zoom.pdf", fig; backend = CairoMakie)
    fig
end

# save_object("$outdir/burgers_series.jld2", series)
# series = load_object("$outdir/burgers_series.jld2")

# Compute relative errors
errseries = let
    fields = map([:nomodel, :class_m, :class_p, :swapfil]) do key
        e = map(series) do (; nH, fields)
            (; u) = fields[key]
            norm(u - fields.dns_fil.u) / norm(fields.dns_fil.u)
        end
        key => (; e, series[1].fields[key].label)
    end
    (; nH = getindex.(series, :nH), fields = (; fields...))
end;
errseries.fields |> pairs

# Write errors to LaTeX table
let
    path = joinpath(outdir, "tables") |> mkpath
    file = joinpath(path, "burgers_error.tex")
    open(file, "w") do io
        tab = "    "
        c = join(fill("r", length(errseries.fields) + 1), " ")
        println(io, "\\begin{tabular}{$c}")
        println(io, tab, "\\toprule")
        labels = join(map(f -> f.label, errseries.fields), " & ")
        println(io, tab, "N & $labels \\\\")
        println(io, tab, "\\midrule")
        for i in eachindex(errseries.nH)
            e = map(f -> round(f.e[i]; sigdigits = 3), errseries.fields)
            println(io, tab, errseries.nH[i], " & ", join(e, " & "), " \\\\")
        end
        println(io, tab, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io)
        println(io, "% vim: conceallevel=0 textwidth=0")
    end
    open(readlines, file) .|> println
    nothing
end

# Compute spectra
specseries = map(series) do (; nH, fields)
    @show nH
    gh = Grid(setup.L, setup.nh)
    gH = Grid(setup.L, nH)
    specs = map(fields) do (; u, g, label)
        uhat = rfft(u, 1)
        n, nsample = size(u)
        e = sum(u -> abs2(u) / 2 / n^2 / nsample, uhat; dims = 2)
        e = e[2:end]
        k = 1:div(n, 2)
        (; k, e, label)
    end
    (; nH, specs)
end;

# Plot spectra
let
    fig = Figure(; size = (400, 800))
    f = fig[1, 1] = GridLayout()
    ticks = 2 .^ (0:2:10)
    axes = map(specseries[1:end] |> enumerate) do (i, specseries)
        (; nH, specs) = specseries
        islast = i == 3
        ax = Axis(
            f[i, 1];
            yscale = log10,
            xscale = log10,
            xlabel = "Wavenumber",
            ylabel = "Energy",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        tip = specs.dns_fil
        # o = (22, 70, 210)[i]
        o = (22, 38, 80)[i]
        ax_zoom = ExactClosure.zoombox!(
            f[i, 1],
            ax;
            point = (tip.k[end-o], tip.e[end-o]),
            logx = 1.3,
            logy = 2.5,
            relwidth = 0.45,
            relheight = 0.45,
        )
        styles = (;
            dns_ref = (; color = :black),
            dns_fil = (; color = :black, linestyle = :dash),
            nomodel = (; color = Cycled(1)),
            class_m = (; color = Cycled(2)),
            class_p = (; color = Cycled(3)),
            swapfil = (; color = Cycled(4)),
        )
        for key in [:dns_ref, :nomodel, :class_m, :class_p, :swapfil, :dns_fil]
            (; k, e, label) = specs[key]
            # At the end of the spectrum, there are too many points for plotting.
            # Choose a logarithmically equispaced subset of points instead
            npoint = 500
            ii = round.(Int, logrange(1, length(k), npoint)) |> unique
            lines!(ax, k[ii], e[ii]; label, styles[key]...)
            lines!(ax_zoom, k[ii], e[ii]; styles[key]...)
        end
        Label(
            f[i, 1],
            "N = $nH";
            # fontsize = 26,
            font = :bold,
            padding = (10, 10, 10, 10),
            halign = :right,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
        ax
    end
    Legend(
        fig[0, 1],
        axes[1];
        tellwidth = false,
        orientation = :horizontal,
        nbanks = 2,
        framevisible = false,
    )
    rowgap!(fig.layout, 10)
    save("$(plotdir)/burgers_spectrum.pdf", fig; backend = CairoMakie)
    fig
end

# Compute dissipation coefficients
diss = compute_dissipation(series, setup, :gaussian)

# Plot dissipation coefficient density
let
    models = [
        (; label = "No-model", sym = :nomodel),
        (; label = "Classic", sym = :classic),
        (; label = "Swap (ours)", sym = :swapfil),
    ]
    fig = Figure(; size = (400, 800))
    for (i, diss) in diss |> enumerate
        ax = Axis(
            fig[i, 1];
            xticks = (eachindex(models), getindex.(models, :label)),
            xticksvisible = i == 3,
            xticklabelsvisible = i == 3,
            ylabel = "Dissipation",
        )
        for (i, model) in enumerate(models)
            d = diss[model.sym]
            boxplot!(
                ax,
                fill(i, length(d)),
                d;
                show_outliers = false,
                whiskerwidth = 0.2,
                orientation = :vertical,
                label = model[1],
                color = Cycled(i + 1),
            )
        end
        Label(
            fig[i, 1],
            "N = $(setup.nH[i])";
            font = :bold,
            padding = (10, 0, 0, 10),
            halign = :left,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
    end
    save("$(plotdir)/burgers_dissipation.pdf", fig; backend = CairoMakie)
    fig
end

# Plot dissipation coefficient density
let
    models = [
        (; label = "No-model", sym = :nomodel),
        (; label = "Classic", sym = :class_m),
        (; label = "Classic+", sym = :class_p),
        (; label = "Swap (ours)", sym = :swapfil),
    ]
    fig = Figure(; size = (400, 800))
    for (i, diss) in diss |> enumerate
        n = setup.nH[i]
        g = Grid(setup.L, n)
        Δx = spacing(g)
        Δ = setup.Δ_ratio * Δx
        a = 1.0e2
        ax = Axis(
            fig[i, 1];
            xlabelvisible = i == 3,
            xticksvisible = i == 3,
            xticklabelsvisible = i == 3,
            xlabel = "Dissipation",
            ylabel = "Density",
            yscale = log10,
        )
        xlims!(ax, -0.5 * a, 0.5 * a)
        ylims!(ax, 1e-4, 1.0e-1)
        for (j, model) in enumerate(models)
            d = diss[model.sym] / (Δ^2 + Δx^2)
            s = median(d)
            if model.sym == :nomodel
                lines!(
                    ax,
                    [Point2(s, 1e-5), Point2(s, 1e0)];
                    color = Cycled(j),
                    model.label,
                )
            else
                # lines!(ax, [Point2(s, 1e-5), Point2(s, 1e0)]; color = Cycled(j), linestyle = :dash)
                k = kde(d; boundary = (-a, a))
                lines!(ax, k.x, k.density; model.label, color = Cycled(j))
            end
        end
        Label(
            fig[i, 1],
            "N = $(setup.nH[i])";
            font = :bold,
            padding = (10, 0, 0, 10),
            halign = :left,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
        i == 1 && Legend(
            fig[0, 1],
            ax;
            tellwidth = false,
            orientation = :horizontal,
            framevisible = false,
            nbanks = 2,
        )
    end
    # rowgap!(fig.layout, 10)
    file = "$(plotdir)/burgers_dissipation.pdf"
    @info "Saving dissipation plot to $file"
    save(file, fig; backend = CairoMakie)
    fig
end
