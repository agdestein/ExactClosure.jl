if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

@info "Loading packages"
flush(stderr)

using CairoMakie
using JLD2
using StructuralClosure: NavierStokes, zoombox!
using WGLMakie

@info "Loading case"
flush(stderr)

case = NavierStokes.largecase()
case = NavierStokes.newcase()

(; grid, outdir, datadir, plotdir, n_les, compression) = case
@assert all(==(grid.n), n_les .* compression)

# Horizontal
let
    fig = Figure(; size = (900, 400))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, n_les) in enumerate(n_les), (j, experiment) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        relerr = load(joinpath(datadir, "relerr-$(experiment)-$(n_les).jld2"), "relerr")
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
            "N = $(n_les), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[experiment];
            # Dict(
            #     "volavg" => "Volume-average",
            #     "project_volavg" => "Projected volume-average",
            #     "surfavg" => "Surface-average",
            # )[experiment];
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

# Vertical
let
    fig = Figure(; size = (480, 550))
    for (i, experiment) in enumerate(["volavg", "project_volavg", "surfavg"]),
        (j, n_les) in enumerate(n_les)

        ilast = i == 3
        jfirst = j == 1
        relerr = load(joinpath(datadir, "relerr-$(experiment)-$(n_les).jld2"), "relerr")
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
            fig[0, 1:2],
            ax;
            tellwidth = false,
            orientation = :horizontal,
            nbanks = 1,
            framevisible = false,
        )
        Label(
            fig[i, j],
            "N = $(n_les), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[experiment];
            # Dict(
            #     "volavg" => "Volume-average",
            #     "project_volavg" => "Projected volume-average",
            #     "surfavg" => "Surface-average",
            # )[experiment];
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
    file = joinpath(plotdir, "ns-error-vert.pdf")
    @info "Saving error plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# Plot spectrum
# Horizontal
let
    fig = Figure(; size = (900, 400))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, n_les) in enumerate(n_les), (j, experiment) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        specs, D =
            load(joinpath(datadir, "spectra-$(experiment)-$(n_les).jld2"), "specs", "D")
        xslope = specs.dns_ref.k[8:(end-12)]
        yslope = @. 1.58 * D^(2 / 3) * (xslope)^(-5 / 3)
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
            logx = 1.2,
            logy = 1.8,
            relwidth = 0.45,
            relheight = 0.55,
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
        # lines!(ax_full, xslope, yslope; color = Cycled(1), linestyle = :dot, label = "Kolmorov")
        # lines!(ax_zoom, xslope, yslope; color = Cycled(1), linestyle = :dot)
        # o = 12
        # text!(ax_full, xslope[end-o], 1.3 * yslope[end-o]; color = Makie.wong_colors()[1], text = "Kolmorov")
        # axislegend(ax_full; position = :lb)
        Label(
            fig[i, j],
            "N = $(n_les), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[experiment];
            # Dict(
            #     "volavg" => "Volume-average",
            #     "project_volavg" => "Projected volume-average",
            #     "surfavg" => "Surface-average",
            # )[experiment];
            # fontsize = 26,
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
    # vlines!(ax_full, [g_dns.n / 2, g_les.n / 2])
    # vlines!(ax_full, 150)
    # ylims!(ax_full, 1e-4, 1e-1)
    # rowgap!(fig.layout, 5)
    file = joinpath(plotdir, "ns-spectra.pdf")
    @info "Saving spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# Plot spectrum
# Vertical
let
    fig = Figure(; size = (450, 500))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, experiment) in enumerate(experiments), (j, n_les) in enumerate(n_les)
        ilast = i == 3
        jfirst = j == 1
        specs, D =
            load(joinpath(datadir, "spectra-$(experiment)-$(n_les).jld2"), "specs", "D")
        xslope = specs.dns_ref.k[8:(end-12)]
        yslope = @. 1.58 * D^(2 / 3) * (2π * xslope)^(-5 / 3)
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
            logx = 1.2,
            logy = 1.8,
            relwidth = 0.45,
            relheight = 0.55,
        )
        for (key, label, color, linestyle) in [
            (:nomodel, "No-model", Cycled(2), :solid),
            (:classic, "Classic", Cycled(3), :solid),
            (:swapfil, "Swap", Cycled(4), :solid),
            (:swapfil_symm, "Swap-sym", Cycled(5), :solid),
            (:dns_fil, "Filtered DNS", Cycled(1), :dash),
            (:kolmogo, "Kolmogorov", Cycled(1), :dot),
            (:dns_ref, "DNS", Cycled(1), :solid),
        ]
            spec = specs[key]
            lines!(ax_full, Point2f.(spec.k, spec.s); color, linestyle, label)
            lines!(ax_zoom, Point2f.(spec.k, spec.s); color, linestyle)
        end
        # lines!(ax_full, xslope, yslope; color = Cycled(1), linestyle = :dot, label = "Kolmorov")
        # lines!(ax_zoom, xslope, yslope; color = Cycled(1), linestyle = :dot)
        # o = 12
        # text!(ax_full, xslope[end-o], 1.3 * yslope[end-o]; color = Makie.wong_colors()[1], text = "Kolmorov")
        # axislegend(ax_full; position = :lb)
        Label(
            fig[i, j],
            "N = $(n_les), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[experiment];
            # Dict(
            #     "volavg" => "Volume-average",
            #     "project_volavg" => "Projected volume-average",
            #     "surfavg" => "Surface-average",
            # )[experiment];
            # fontsize = 26,
            font = :bold,
            padding = (7, 7, 7, 7),
            halign = :right,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
        (i, j) == (1, 1) && Legend(
            fig[0, 1:2],
            ax_full;
            tellwidth = false,
            tellheight = true,
            framevisible = false,
            orientation = :horizontal,
            nbanks = 2,
        )
    end
    # vlines!(ax_full, [g_dns.n / 2, g_les.n / 2])
    # vlines!(ax_full, 150)
    # ylims!(ax_full, 1e-4, 1e-1)
    # rowgap!(fig.layout, 5)
    # colgap!(fig.layout, 5)
    file = joinpath(plotdir, "ns-spectra-vert.pdf")
    @info "Saving spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# Dissipation
let
    fig = Figure(; size = (900, 400))
    experiments = ["volavg", "project_volavg", "surfavg"]
    for (i, n_les) in enumerate(n_les), (j, experiment) in enumerate(experiments)
        ilast = i == 2
        jfirst = j == 1
        file = joinpath(datadir, "dissipation-$(experiment)-$(n_les).jld2")
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
        # plot(1, 3e-2 * randn(length(diss.classic.data)), "No-model")
        plot(2, diss.classic, "Classic")
        plot(3, diss.swapfil_symm, "Swap-sym")
        plot(4, diss.swapfil, "Swap")
        (i, j) == (1, 1) &&
            Legend(fig[1:2, 4], ax; tellheight = false, framevisible = false)
        Label(
            fig[i, j],
            "N = $(n_les), " *
            Dict("volavg" => "VA", "project_volavg" => "PVA", "surfavg" => "SA")[experiment];
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
