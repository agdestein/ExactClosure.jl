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
using KernelDensity
using LinearAlgebra
using Random
using StructuralClosure: NavierStokes
using Turbulox
using Turbulox.KernelAbstractions
using WGLMakie
using WriteVTK

@info "Loading case"
flush(stderr)

begin
    case = NavierStokes.smallcase()
    n_les = 50
    compression = 3
end

begin
    case = NavierStokes.largecase()
    n_les = 100
    compression = 5
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

# plotdir = "~/Projects/StructuralErrorPaper/figures" |> expanduser

poisson_dns = poissonsolver(g_dns);
poisson_les = poissonsolver(g_les);

ustart = let
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    VectorField(g_dns, data)
end

s = get_scale_numbers(ustart, viscosity)
s |> pairs
1 / s.λ

false && let
    uh = ustart
    uH = VectorField(g_les)
    # Turbulox.surfacefilter!(uH, uh, compression)
    Turbulox.volumefilter!(uH, uh, compression)
    uh.data[:, :, end, 3] |> Array |> heatmap
    # uH.data[:, :, end, 3] |> Array |> heatmap
    # nothing
end

false && let
    u = ustart
    fu = VectorField(g_dns)
    Turbulox.volumefilter_explicit!(fu, u, compression)
    # u.data[:, :, 1, 1] |> Array |> heatmap
    fu.data[:, :, 1, 1] |> Array |> heatmap
    # u.data[:, :, 1, 1] - fu.data[:, :, 1, 1] |> Array |> heatmap
end

false && let
    @info "Computing Q-criterion"
    flush(stderr)
    u = ustart
    (; grid) = u
    n = div(compression, 2) + 1
    fu = VectorField(grid)
    unorm = ScalarField(grid)
    funorm = ScalarField(grid)
    qu = ScalarField(grid)
    qfu = ScalarField(grid)
    qfu_coarse = ScalarField(grid)
    Turbulox.volumefilter_explicit!(fu, u, compression)
    apply!(Turbulox.velocitynorm!, grid, unorm, u)
    apply!(Turbulox.velocitynorm!, grid, funorm, fu)
    ∇u = LazyScalarField(grid, Turbulox.∇_collocated, 1, u)
    ∇fu = LazyScalarField(grid, Turbulox.∇_collocated, 1, fu)
    ∇fu_coarse = LazyScalarField(grid, Turbulox.∇_collocated, n, fu)
    apply!(Turbulox.compute_q!, grid, qu, ∇u)
    apply!(Turbulox.compute_q!, grid, qfu, ∇fu)
    apply!(Turbulox.compute_q!, grid, qfu_coarse, ∇fu_coarse)
    x = get_axis(u.grid, Coll())
    file = joinpath(outdir, "data", "q")
    @info "Writing Q to $file"
    flush(stderr)
    vtk_grid(file, x, x, x) do vtk
        vtk["Qh(u)"] = qu.data |> Array
        vtk["Qh(ubar)"] = qfu.data |> Array
        vtk["QH(ubar)"] = qfu_coarse.data |> Array
        vtk["norm(u)"] = unorm.data |> Array
        vtk["norm(ubar)"] = funorm.data |> Array
    end
end

false && let
    u = ustart
    ru = TensorField(g_dns)
    dru = ScalarField(g_dns)
    apply!(stresstensor!, g_dns, ru, u, viscosity)
    apply!(Turbulox.tensordissipation_staggered!, g_dns, dru, ru, u)
    dru.data
    boxplot(
        fill(1, length(dru)),
        dru.data |> vec |> Array;
        show_outliers = false,
        whiskerwidth = 0.2,
        orientation = :vertical,
    )
end

# boxplot(
#     fill(1, 1),
#     zeros(1);
#     show_outliers = false,
#     whiskerwidth = 0.2,
#     orientation = :vertical,
# )

@info "Computing SFS tensors"
flush(stderr)

experiment = "volavg"
sfs = NavierStokes.sfs_tensors_volume(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject = false,
);

experiment = "project_volavg"
sfs = NavierStokes.sfs_tensors_volume(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject = true,
);

experiment = "surfavg"
sfs = NavierStokes.sfs_tensors_surface(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject = false,
);

# plotdir = "~/Projects/StructuralErrorPaper/figures/$experiment" |> expanduser |> mkpath

true && let
    u = ustart
    fu = VectorField(g_les)
    if experiment == "volavg"
        volumefilter!(fu, u, compression)
    elseif experiment == "project_volavg"
        volumefilter!(fu, u, compression)
        p = ScalarField(g_les)
        project!(fu, p, poisson_les)
    elseif experiment == "surfavg"
        surfacefilter!(fu, u, compression)
    else
        error("Unknown experiment: $experiment")
    end
    #
    xy = ScalarField(g_les)
    @kernel function getfrac!(xy, σ, σ_symm)
        I = @index(Global, Cartesian)
        x, y, z = X(), Y(), Z()
        xy[I] = sqrt(2 * abs2(σ_symm[x, y][I]) / (abs2(σ[x, y][I]) + abs2(σ[y, x][I])))
    end
    apply!(getfrac!, g_les, xy, sfs.σ_swapfil, sfs.σ_swapfil_symm)
    xy.data
    frac = 1 - sum(xy.data) / length(xy.data)
    round(frac; sigdigits = 3)
end

# Volavg: 0.101
# Proj.vol: 0.1
# Surfavg: 0.0

false && let
    x, y, z = X(), Y(), Z()
    i, j = x, x
    sfs.classic[i, j].data[1, :, :] |> Array |> heatmap
    sfs.swapfil[i, j].data[1, :, :] |> Array |> heatmap
    # sfs.classic[i, j].data[:, :, 1]
    # sfs.swapfil[i, j].data[:, :, 1]
    # norm(1.0*sfs.classic.data - sfs.swapfil.data) / norm(sfs.swapfil.data)
end

let
    u = ustart
    # u = sols.dns_ref
    fu = VectorField(g_les)
    if experiment == "volavg"
        volumefilter!(fu, u, compression)
    elseif experiment == "project_volavg"
        volumefilter!(fu, u, compression)
        p = ScalarField(g_les)
        project!(fu, p, poisson_les)
    elseif experiment == "surfavg"
        surfacefilter!(fu, u, compression)
    else
        error("Unknown experiment: $experiment")
    end
    diss = (;
        classic = ScalarField(g_les),
        swapfil = ScalarField(g_les),
        swapfil_symm = ScalarField(g_les),
        # rfu = ScalarField(g_les),
    )
    apply!(Turbulox.tensordissipation_staggered!, g_les, diss.classic, sfs.σ_classic, fu)
    apply!(Turbulox.tensordissipation_staggered!, g_les, diss.swapfil, sfs.σ_swapfil, fu)
    apply!(
        Turbulox.tensordissipation_staggered!,
        g_les,
        diss.swapfil_symm,
        sfs.σ_swapfil_symm,
        fu,
    )
    # apply!(Turbulox.tensordissipation_staggered!, g_les, diss.rfu, sfs.rfu, fu)
    diss = adapt(Array, diss)
    jldsave(joinpath(datadir, "$(experiment)_diss.jld2"); diss)
end

true && let
    fig = Figure(; size = (400, 300))
    # ax = Axis(fig[1, 1]; xticks = ([1, 2], ["Classic", "Filter-swap"]))
    ax = Axis(fig[1, 1]; xticks = (1:4, ["No-model", "Classic", "Swap-sym", "Swap"]))
    # ax = Axis(fig[1, 1]; xticks = ([1, 2, 3], ["Classic", "Filter-swap", "Stress"]))
    plot(i, d) = boxplot!(
        ax,
        fill(i, length(d)),
        d;
        show_outliers = false,
        whiskerwidth = 0.2,
        orientation = :vertical,
    )
    plot(1, zeros(1))
    plot(2, diss.classic.data |> vec |> Array)
    plot(3, diss.swapfil_symm.data |> vec |> Array)
    plot(4, diss.swapfil.data |> vec |> Array)
    # plot(5, diss.rfu.data |> vec |> Array)
    file = joinpath(plotdir, "$(experiment)-ns-dissipation.pdf")
    @info "Saving dissipation plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    # ylims!(ax, -0.0005, 0.0005)
    fig
end

true && let
    fig = Figure(; size = (410, 750))
    for (i, experiment) in enumerate(["volavg", "project_volavg", "surfavg"])
        islast = i == 3
        diss = load(joinpath(datadir, "$(experiment)_diss.jld2"), "diss")
        ax = Axis(
            fig[i, 1];
            yscale = log10,
            xlabel = "Dissipation",
            ylabel = "Density",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        xlims!(ax, -17, 12)
        ylims!(ax, 1e-5, 4e-1)
        plot(i, d, label) = hist!(
            ax,
            d;
            # bins = 100,
            bins = range(-17, 12, 100),
            normalization = :probability,
            # strokewidth = 1,
            # color = Cycled(i),
            color = Makie.wong_colors()[i],
            label,
        )
        # plot(d) = density!(ax, d; npoints = 500, strokewidth = 1)
        plot(4, diss.swapfil.data |> vec |> Array, "Swap")
        plot(3, diss.swapfil_symm.data |> vec |> Array, "Swap-sym")
        plot(2, diss.classic.data |> vec |> Array, "Classic")
        # plot(4, 3e-2 * randn(length(diss.classic.data)), "No-model")
        vlines!(ax, [0]; color = Cycled(1), label = "No-model")
        # plot(5, diss.rfu.data |> vec |> Array)
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
    file = joinpath(plotdir, "ns-dissipation-hist.pdf")
    @info "Saving dissipation plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    # ylims!(ax, -0.0005, 0.0005)
    fig
end

true && let
    fig = Figure(; size = (410, 750))
    for (i, experiment) in enumerate(["volavg", "project_volavg", "surfavg"])
        islast = i == 3
        diss = load(joinpath(datadir, "$(experiment)_diss.jld2"), "diss")
        ax = Axis(
            fig[i, 1];
            yscale = log10,
            xlabel = "Dissipation",
            ylabel = "Density",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        # xlims!(ax, -17, 12)
        # ylims!(ax, 1e-4, 1e0)
        # xlims!(ax, -15, 11)
        xlims!(ax, -13, 9)
        ylims!(ax, 7e-5, 2e0)
        function plot(i, d, label)
            k = kde(d)#; boundary = (-8, 8))
            lines!(
                ax,
                k.x,
                k.density;
                color = Cycled(i),
                # color = Makie.wong_colors()[i],
                label,
            )
        end
        # plot(i, d, label) = hist!(
        #     ax,
        #     d;
        #     # bins = 100,
        #     bins = range(-17, 12, 100),
        #     normalization = :probability,
        #     # strokewidth = 1,
        #     # color = Cycled(i),
        #     color = Makie.wong_colors()[i],
        #     label,
        # )
        # plot(d) = density!(ax, d; npoints = 500, strokewidth = 1)
        vlines!(ax, [0]; color = Cycled(1), label = "No-model")
        # plot(1, 3e-2 * randn(length(diss.classic.data)), "No-model")
        plot(2, diss.classic.data |> vec |> Array, "Classic")
        plot(3, diss.swapfil_symm.data |> vec |> Array, "Swap-sym")
        plot(4, diss.swapfil.data |> vec |> Array, "Swap")
        # plot(5, diss.rfu.data |> vec |> Array)
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
    file = joinpath(plotdir, "ns-dissipation.pdf")
    @info "Saving dissipation plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    # ylims!(ax, -0.0005, 0.0005)
    fig
end
