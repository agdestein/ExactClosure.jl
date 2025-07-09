if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA
using JLD2
using LinearAlgebra
using Printf
using Random
using StructuralClosure: NavierStokes, zoombox!
using Turbulox
using WGLMakie

@info "Loading case"
flush(stderr)

begin
    case = NavierStokes.smallcase()
    n_les = 50
    compression = 3
end

begin
    case = NavierStokes.largecase()
    # n_les, compression = 102, 5
    n_les, compression = 170, 3
end

begin
    case = NavierStokes.snelliuscase()
    n_les = 160
    compression = 5
end

(; viscosity, outdir, datadir, plotdir, seed) = case
g_dns = case.grid
g_les = Grid(; g_dns.ho, g_dns.backend, g_dns.L, n = n_les)
T = typeof(g_dns.L)
@assert n_les * compression == g_dns.n

poisson_dns = poissonsolver(g_dns);
poisson_les = poissonsolver(g_les);

# ustart = let
#     path = joinpath(outdir, "u.jld2")
#     data = path |> load_object |> adapt(g_dns.backend)
#     VectorField(g_dns, data)
# end

let
    cfl = 0.15 |> T
    tstop = 0.001 |> T
    # Load initial DNS
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    ustart = VectorField(g_dns, data)
    for experiment in ["volavg", "project_volavg", "surfavg"]
        @info "Running experiment: $(experiment)"
        flush(stderr)
        if experiment == "volavg"
            sols, relerr = NavierStokes.dns_aid_volavg(;
                ustart,
                g_dns,
                g_les,
                poisson_dns,
                poisson_les,
                viscosity,
                compression,
                doproject = false,
                docopy = false,
                tstop,
                cfl,
            )
        elseif experiment == "project_volavg"
            sols, relerr = NavierStokes.dns_aid_volavg(;
                ustart,
                g_dns,
                g_les,
                poisson_dns,
                poisson_les,
                viscosity,
                compression,
                doproject = true,
                docopy = false,
                tstop,
                cfl,
            )
        elseif experiment == "surfavg"
            sols, relerr = NavierStokes.dns_aid_surface(;
                ustart,
                g_dns,
                g_les,
                poisson_dns,
                poisson_les,
                viscosity,
                compression,
                doproject = true,
                docopy = false,
                tstop,
                cfl,
            )
        end
        # Save errors
        file = joinpath(datadir, "relerr-$(experiment)-$(n_les).jld2")
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
        file = joinpath(datadir, "spectra-$(experiment)-$(n_les).jld2")
        @info "Saving spectra to $(file)"
        flush(stderr)
        jldsave(file; specs, D)
        sols = nothing # free up memory
    end
end

let
    fig = Figure(; size = (410, 750))
    for (i, experiment) in enumerate(["volavg", "project_volavg", "surfavg"])
        islast = i == 3
        relerr = load(joinpath(datadir, "relerr-$(experiment)-$(n_les).jld2"), "relerr")
        ax = Axis(
            fig[i, 1];
            xlabel = "Time",
            ylabel = "Relative error",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        lines!(ax, relerr.time, relerr.nomodel; label = "No-model")
        lines!(ax, relerr.time, relerr.classic; label = "Classic")
        lines!(ax, relerr.time, relerr.swapfil_symm; label = "Swap-sym")
        lines!(ax, relerr.time, relerr.swapfil; label = "Swap")
        # axislegend(ax; position = :lt, framevisible = true)
        i == 1 && Legend(
            fig[0, 1],
            ax;
            tellwidth = false,
            orientation = :horizontal,
            # nbanks = 2,
            framevisible = false,
        )
        Label(
            fig[i, 1],
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
    file = joinpath(plotdir, "ns-error-$(n_les).pdf")
    @info "Saving error plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end

# Plot spectrum
let
    fig = Figure(; size = (400, 750))
    for (i, experiment) in enumerate(["volavg", "project_volavg", "surfavg"])
        islast = i == 3
        specs, D =
            load(joinpath(datadir, "spectra-$(experiment)-$(n_les).jld2"), "specs", "D")
        xslope = specs.dns_ref.k[8:(end-12)]
        yslope = @. 1.58 * D^(2 / 3) * (2π * xslope)^(-5 / 3)
        specs = (; specs..., kolmogo = (; k = xslope, s = yslope))
        ax_full = Makie.Axis(
            fig[i, 1];
            xscale = log2,
            yscale = log10,
            xlabel = "Wavenumber",
            ylabel = "Energy",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        o = 10
        ax_zoom = zoombox!(
            fig[i, 1],
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
            fig[i, 1],
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
        i == 1 && Legend(
            fig[0, 1],
            ax_full;
            tellwidth = false,
            orientation = :horizontal,
            nbanks = 3,
            framevisible = false,
        )
    end
    # vlines!(ax_full, [g_dns.n / 2, g_les.n / 2])
    # vlines!(ax_full, 150)
    # ylims!(ax_full, 1e-4, 1e-1)
    # rowgap!(fig.layout, 5)
    file = joinpath(plotdir, "ns-spectra-$(n_les).pdf")
    @info "Saving spectrum plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end
