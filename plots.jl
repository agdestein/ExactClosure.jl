# This is just a hack for "go to definition" to work in editor.
if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using JLD2
using StructuralClosure: NavierStokes, zoombox!
using Turbulox
using WGLMakie

@info "Loading case"
flush(stderr)

# Choose case
case = NavierStokes.smallcase() # Laptop CPU with 16 GB RAM
case = NavierStokes.mediumcase() # GPU with 24 GB RAM
case = NavierStokes.largecase() # GPU with 90 GB RAM (H100)

(; grid, outdir, datadir, plotdir, g_dns, g_les) = case

# Plot DNS and filtered DNS in zoom-in region
let
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    u = VectorField(g_dns, data)
    ubar = map(g_les) do g_les
        compression = div(g_dns.n, g_les.n)
        v = VectorField(g_les)
        Turbulox.volumefilter!(v, u, compression)
        v
    end
    fig = Figure(; size = (860, 300))
    kwargs = (;
        xlabelvisible = false,
        ylabelvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        aspect = DataAspect(),
    )
    A = 90
    imkwargs = (; colormap = :seaborn_icefire_gradient, interpolate = false)
    i = 1
    image!(
        Axis(fig[1, 1]; kwargs...),
        u.data[(end-A+1):end, (end-A+1):end, end, i] |> Array;
        imkwargs...,
    )
    for (j, ubar) in enumerate(ubar)
        compression = div(g_dns.n, ubar.grid.n)
        a = div(A, compression)
        image!(
            Axis(fig[1, 1+(length(ubar)+1-j)]; kwargs...),
            ubar.data[(end-a+1):end, (end-a+1):end, end, i] |> Array;
            imkwargs...,
        )
    end
    file = joinpath(plotdir, "ns-fields-zoom.png")
    @info "Saving fields plot to $file"
    save(file, fig; backend = CairoMakie)
    fig
end

# Relative errors
let
    fig = Figure(; size = (900, 400))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, g_les) in enumerate(g_les), (j, ex) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        relerr = load(joinpath(datadir, "relerr-$(ex)-$(g_les.n).jld2"), "relerr")
        ax = Axis(
            fig[i, j];
            xlabel = "Time",
            ylabel = "Relative error",
            xticksvisible = ilast,
            xticklabelsvisible = ilast,
            xlabelvisible = ilast,
            yticksvisible = jfirst,
            yticklabelsvisible = jfirst,
            ylabelvisible = jfirst,
        )
        ylims!(ax, -0.04, 0.73)
        lines!(ax, relerr.time, relerr.nomodel; label = "No-model")
        lines!(ax, relerr.time, relerr.classic; label = "Classic")
        lines!(ax, relerr.time, relerr.swapfil_symm; label = "Swap-sym")
        lines!(ax, relerr.time, relerr.swapfil; label = "Swap")
        # axislegend(ax; position = :lt, framevisible = true)
        (i, j) == (1, 1) && Legend(
            fig[1:2, 4],
            ax;
            tellwidth = true,
            tellheight = false,
            # orientation = :horizontal,
            # nbanks = 2,
            framevisible = false,
        )
        Label(
            fig[i, j],
            "N = $(g_les.n), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[ex];
            # Dict(
            #     "volavg" => "Volume-average",
            #     "project_volavg" => "Projected volume-average",
            #     "surfavg" => "Surface-average",
            # )[ex];
            # fontsize = 26,
            font = :bold,
            padding = (10, 10, 10, 10),
            halign = :left,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
    end
    # rowgap!(fig.layout, 5)
    # rowgap!(fig.layout, 10)
    # ylims!(ax, -0.03, 0.34)
    file = joinpath(plotdir, "ns-error.pdf")
    @info "Saving error plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# let
#     experiments = ["volavg", "project_volavg", "surfavg"]
#     for g_les in g_les, ex in experiments
#         # run(`echo $("spectra-$(ex)-$(g_les.n).jld2")`)
#         inpath = "snellius:/projects/0/prjs0936/syver/ExactClosure/newcase/data/spectra-$(ex)-$(g_les.n).jld2"
#         outpath = "output/newcase/data/"
#         run(`scp $inpath $outpath`)
#     end
# end

# Plot spectrum
let
    fig = Figure(; size = (900, 400))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, g_les) in enumerate(g_les), (j, ex) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        specs, D =
            load(joinpath(datadir, "spectra-$(ex)-$(g_les.n).jld2"), "specs", "D")
        xslope = specs.dns_ref.k[8:(end-12)]
        CK = 0.5
        yslope = @. CK * D^(2 / 3) * (xslope)^(-5 / 3)
        specs = (; specs..., kolmogo = (; k = xslope, s = yslope))
        ax_full = Makie.Axis(
            fig[i, j];
            xscale = log10,
            yscale = log10,
            xlabel = "Wavenumber",
            ylabel = "Energy",
            xticksvisible = ilast,
            xticklabelsvisible = ilast,
            xlabelvisible = ilast,
            yticksvisible = jfirst,
            yticklabelsvisible = jfirst,
            ylabelvisible = jfirst,
        )
        o = 10
        ax_zoom = zoombox!(
            fig[i, j],
            ax_full;
            point = (specs.dns_fil.k[end-o], specs.dns_fil.s[end-o]),
            logx = 1.3,
            logy = 2.2,
            relwidth = 0.5,
            relheight = 0.65,
        )
        for (key, label, color, linestyle) in [
            (:nomodel, "No-model", Cycled(2), :solid),
            (:classic, "Classic", Cycled(3), :solid),
            (:swapfil, "Swap", Cycled(4), :solid),
            (:swapfil_symm, "Swap-sym", Cycled(5), :solid),
            (:dns_ref, "DNS", Cycled(1), :solid),
            (:dns_fil, "Filtered DNS", Cycled(1), :dash),
            (:kolmogo, "Kolmogorov", Cycled(1), :dot),
        ]
            spec = specs[key]
            lines!(ax_full, Point2f.(spec.k, spec.s); color, linestyle, label)
            lines!(ax_zoom, Point2f.(spec.k, spec.s); color, linestyle)
        end
        Label(
            fig[i, j],
            "N = $(g_les.n), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[ex];
            font = :bold,
            padding = (10, 10, 10, 10),
            halign = :right,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
        (i, j) == (1, 1) && Legend(
            fig[1:2, 4],
            ax_full;
            tellwidth = true,
            tellheight = false,
            framevisible = false,
        )
    end
    file = joinpath(plotdir, "ns-spectra.pdf")
    @info "Saving spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# Dissipation
let
    fig = Figure(; size = (900, 400))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, g_les) in enumerate(g_les), (j, ex) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        file = joinpath(datadir, "dissipation-$(ex)-$(g_les.n).jld2")
        diss = load(file, "density")
        ax = Axis(
            fig[i, j];
            yscale = log10,
            xlabel = "Dissipation",
            ylabel = "Density",
            xticksvisible = ilast,
            xticklabelsvisible = ilast,
            xlabelvisible = ilast,
            yticksvisible = jfirst,
            yticklabelsvisible = jfirst,
            ylabelvisible = jfirst,
        )
        # limits!(ax, (-6, 4), (1e-3, 3e0))
        limits!(ax, (-9, 6.3), (1e-4, 3e0))
        plot(i, d, label) = lines!(ax, d.x, d.density; color = Cycled(i), label)
        vlines!(ax, [0]; color = Cycled(1), label = "No-model")
        plot(2, diss.classic, "Classic")
        plot(3, diss.swapfil_symm, "Swap-sym")
        plot(4, diss.swapfil, "Swap")
        (i, j) == (1, 1) &&
            Legend(fig[1:2, 4], ax; tellheight = false, framevisible = false)
        Label(
            fig[i, j],
            "N = $(g_les.n), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[ex];
            font = :bold,
            padding = (10, 10, 10, 10),
            halign = :left,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
    end
    file = joinpath(plotdir, "ns-dissipation.pdf")
    @info "Saving dissipation plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end
