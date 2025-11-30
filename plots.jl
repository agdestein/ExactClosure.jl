# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using JLD2
using ExactClosure: NavierStokes, zoombox!
using Turbulox
using WGLMakie

@info "Loading case"
flush(stderr)

# Choose case
case = NavierStokes.smallcase() # Laptop CPU with 16 GB RAM
case = NavierStokes.mediumcase() # GPU with 24 GB RAM
case = NavierStokes.largecase() # GPU with 90 GB RAM (H100)

(; outdir, datadir, plotdir, g_dns, g_les) = case

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
    fig = Figure(; size = (700, 240))
    kwargs = (;
        # xlabelvisible = false,
        ylabelvisible = false,
        # xticksvisible = false,
        yticksvisible = false,
        # xticklabelsvisible = false,
        yticklabelsvisible = false,
        # xlabel = "x",
        # ylabel = "y",
        aspect = DataAspect(),
    )
    A = 90
    imkwargs = (; colormap = :seaborn_icefire_gradient, interpolate = false)
    i = 1
    r = (g_dns.n-A):g_dns.n
    bounds = g_dns.L/g_dns.n * (g_dns.n - A), g_dns.L
    dns = u.data[r, r, end, i] |> Array
    im = image!(
        Axis(
            fig[1, 1];
            kwargs...,
            ylabelvisible = true,
            yticksvisible = true,
            yticklabelsvisible = true,
        ),
        bounds,
        bounds,
        dns;
        imkwargs...,
        colorrange = dns |> extrema,
    )
    for (j, v) in enumerate(ubar)
        compression = div(g_dns.n, v.grid.n)
        a = div(A, compression)
        r = (v.grid.n-a):v.grid.n
        bounds = v.grid.L/v.grid.n * (v.grid.n - a), v.grid.L
        image!(
            Axis(fig[1, 1+(length(ubar)+1-j)]; kwargs...),
            bounds,
            bounds,
            v.data[r, r, end, i] |> Array;
            imkwargs...,
            colorrange = dns |> extrema,
        )
    end
    # Colorbar(fig[0, 2:2], im; vertical = false)
    Colorbar(fig[1, 4], im)
    colgap!(fig.layout, 10)
    file = joinpath(plotdir, "ns-fields-zoom.pdf")
    @info "Saving fields plot to $file"
    save(file, fig; backend = CairoMakie)
    fig
end

# Plot relative errors from DNS-aided LES
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

# Plot DNS spectrum after warm-up
let
    spec_dns, spec_les, D =
        load(joinpath(datadir, "warm-up-spectra.jld2"), "dns", "les", "D")
    # Kolmogorov slope
    xslope = spec_dns.k[[8, end-20]]
    CK = 0.5
    yslope = @. CK * D^(2 / 3) * (xslope)^(-5 / 3)
    kolmogo = (; k = xslope, s = yslope)
    fig = Figure(; size = (400, 350))
    ax = Makie.Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xlabel = "Wavenumber",
        ylabel = "Energy",
    )
    colors = [Makie.wong_colors()[1:4]; "#202020"; "#606060"]
    series = [
        (spec_dns, "DNS (N = $(g_dns.n))", colors[1], :solid),
        (spec_les[1], "Filtered DNS (N = $(g_les[1].n))", colors[2], :solid),
        (kolmogo, "Kolmogorov", colors[4], :dash),
        (spec_les[2], "Filtered DNS (N = $(g_les[2].n))", colors[3], :solid),
    ]
    for (s, label, color, linestyle) in series
        lines!(ax, Point2f.(s.k, s.s); color, linestyle, label)
    end
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        tellheight = true,
        framevisible = false,
        horizontal = true,
        nbanks = 2,
    )
    rowgap!(fig.layout, 5)
    file = joinpath(plotdir, "ns-spectrum-warmup.pdf")
    @info "Saving spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# Plot DNS-aided LES spectra
let
    fig = Figure(; size = (900, 400))
    # fig = Figure(; size = (1200, 600))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, g_les) in enumerate(g_les), (j, ex) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        specs, D = load(joinpath(datadir, "spectra-$(ex)-$(g_les.n).jld2"), "specs", "D")
        xslope = specs.dns_ref.k[[8, end-40]]
        CK = 0.5
        yslope = @. CK * D^(2 / 3) * (xslope)^(-5 / 3)
        specs = (; specs..., kolmogo = (; k = xslope, s = yslope))
        ax_full = Makie.Axis(
            fig[i, j];
            xscale = log10,
            yscale = log10,
            xlabel = "Wavenumber",
            ylabel = "Energy",
            xticks = [1, 10, 100],
            xticksvisible = ilast,
            xticklabelsvisible = ilast,
            xlabelvisible = ilast,
            yticksvisible = jfirst,
            yticklabelsvisible = jfirst,
            ylabelvisible = jfirst,
        )
        xlims!(ax_full, 0.8, 1.6e2)
        # ylims!(ax_full, 1.5e-5, 1e-2)
        o = 35
        ax_zoom = zoombox!(
            fig[i, j],
            ax_full;
            point = (specs.dns_fil.k[end-o], specs.dns_fil.s[end-o]),
            logx = 1.15,
            logy = 1.3,
            relwidth = 0.5,
            relheight = 0.55,
        )
        colors = [Makie.wong_colors()[1:4]; "#202020"; "#606060"]
        # series = [
        #     (:dns_ref, "DNS", colors[1], :solid),
        #     (:nomodel, "No-model", colors[2], :solid),
        #     (:classic, "Classic", colors[3], :solid),
        #     (:swapfil, "Swap", colors[4], :solid),
        #     (:swapfil_symm, "Swap-sym", colors[5], :solid),
        #     (:dns_fil, "Filtered DNS", colors[1], :dash),
        #     (:kolmogo, "Kolmogorov", colors[1], :dot),
        # ]
        series = [
            (:nomodel, "No-model", colors[1], :solid),
            (:classic, "Classic", colors[2], :solid),
            (:swapfil_symm, "Swap-sym", colors[3], :solid),
            (:swapfil, "Swap", colors[4], :solid),
            (:dns_fil, "Filtered DNS", colors[5], :dash),
            # (:dns_ref, "DNS", colors[6], :solid),
            (:kolmogo, "Kolmogorov", colors[6], :dot),
        ]
        for (key, label, color, linestyle) in series
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

@info "Writing errors to LaTeX table"
flush(stderr)
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
