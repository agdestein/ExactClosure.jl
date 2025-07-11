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

case = NavierStokes.smallcase()
case = NavierStokes.largecase()
case = NavierStokes.snelliuscase()
case = NavierStokes.newcase()

(; viscosity, outdir, datadir, plotdir, seed, n_les, compression) = case
g_dns = case.grid
g_les = map(n -> Grid(; g_dns.ho, g_dns.backend, g_dns.L, n), n_les);
T = typeof(g_dns.L)

# plotdir = "~/Projects/StructuralErrorPaper/figures" |> expanduser

poisson_dns = poissonsolver(g_dns);
poisson_les = map(g -> poissonsolver(g), g_les);

ustart = let
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    VectorField(g_dns, data)
end

s = get_scale_numbers(ustart, viscosity)
s |> pairs
1 / s.λ
2 / s.λ

# julia> s |> pairs
# pairs(::NamedTuple) with 11 entries:
#   :uavg   => 0.444893
#   :D      => 0.356442
#   :L      => 0.247046
#   :λ      => 0.0037259
#   :eta    => 0.00045757
#   :t_int  => 0.555294
#   :t_tay  => 0.00837482
#   :t_kol  => 0.00837482
#   :Re_int => 4396.37
#   :Re_tay => 66.3052
#   :Re_kol => 8.1428
#
# julia> 1 / s.λ
# 268.3914272838432
# julia> 2 / s.λ
# 536.7828545676864

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

@info "Computing dissipation coefficient density"
flush(stderr)

let
    u = ustart
    for (poisson_les, g_les, n_les, compression) in
        zip(poisson_les, g_les, n_les, compression)
        fu = VectorField(g_les)
        for experiment in ["volavg", "project_volavg", "surfavg"]
            @info "Experiment: $experiment"
            flush(stderr)
            if experiment == "volavg"
                sfs = NavierStokes.sfs_tensors_volume(;
                    ustart,
                    g_dns,
                    g_les,
                    poisson_dns,
                    poisson_les,
                    viscosity,
                    compression,
                    doproject = false,
                )
                volumefilter!(fu, u, compression)
            elseif experiment == "project_volavg"
                sfs = NavierStokes.sfs_tensors_volume(;
                    ustart,
                    g_dns,
                    g_les,
                    poisson_dns,
                    poisson_les,
                    viscosity,
                    compression,
                    doproject = true,
                )
                volumefilter!(fu, u, compression)
                p = ScalarField(g_les)
                project!(fu, p, poisson_les)
            elseif experiment == "surfavg"
                sfs = NavierStokes.sfs_tensors_surface(;
                    ustart,
                    g_dns,
                    g_les,
                    poisson_dns,
                    poisson_les,
                    viscosity,
                    compression,
                    doproject = false,
                )
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
            apply!(
                Turbulox.tensordissipation_staggered!,
                g_les,
                diss.classic,
                sfs.σ_classic,
                fu,
            )
            apply!(
                Turbulox.tensordissipation_staggered!,
                g_les,
                diss.swapfil,
                sfs.σ_swapfil,
                fu,
            )
            apply!(
                Turbulox.tensordissipation_staggered!,
                g_les,
                diss.swapfil_symm,
                sfs.σ_swapfil_symm,
                fu,
            )
            # apply!(Turbulox.tensordissipation_staggered!, g_les, diss.rfu, sfs.rfu, fu)
            diss = adapt(Array, diss)
            density = map(d -> (d.data |> vec |> kde |> x -> (; x.x, x.density)), diss)
            file = joinpath(datadir, "dissipation-$(experiment)-$(n_les).jld2")
            @info "Saving dissipation density to $file"
            flush(stderr)
            jldsave(file; density)
        end
    end
end

let
    u = ustart
    ubar = map(g_les, compression) do g_les, compression
        v = VectorField(g_les)
        Turbulox.volumefilter!(v, u, compression)
        v
    end
    fig = Figure(; size = (800, 800))
    kwargs = (;
        xlabelvisible = false,
        ylabelvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        aspect = DataAspect(),
    )
    imkwargs = (; colormap = :seaborn_icefire_gradient, interpolate = false)
    image!(Axis(fig[1, 1]; kwargs...), u.data[:, :, end, 1] |> Array; imkwargs...)
    image!(Axis(fig[1, 2]; kwargs...), ubar.data[:, :, end, 1] |> Array; imkwargs...)
    file = joinpath(plotdir, "ns-fields-$(n_les).png")
    @info "Saving fields plot to $file"
    save(file, fig; backend = CairoMakie)
    fig
end

let
    u = ustart
    ubar = map(g_les, compression) do g_les, compression
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
    for j = 1:length(ubar)
        v, c = ubar[j], compression[j]
        a = div(A, c)
        image!(
            Axis(fig[1, 1+(length(ubar)+1-j)]; kwargs...),
            v.data[(end-a+1):end, (end-a+1):end, end, i] |> Array;
            imkwargs...,
        )
    end
    file = joinpath(plotdir, "ns-fields-zoom.png")
    @info "Saving fields plot to $file"
    save(file, fig; backend = CairoMakie)
    fig
end

true && let
    n_les = Main.n_les[1]
    fig = Figure(; size = (410, 750))
    for (i, experiment) in enumerate(["volavg", "project_volavg", "surfavg"])
        islast = i == 3
        file = joinpath(datadir, "dissipation-$(experiment)-$(n_les).jld2")
        diss = load(file, "diss")
        ax = Axis(
            fig[i, 1];
            yscale = log10,
            xlabel = "Dissipation",
            ylabel = "Density",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        # xlims!(ax, -13, 9)
        # ylims!(ax, 7e-5, 2e0)
        #
        # xlims!(ax, -13.5, 9)
        # ylims!(ax, 1e-4, 1e0)
        #
        # xlims!(ax, -8.4, 5)
        # ylims!(ax, 1e-3, 1e0)
        #
        # xlims!(ax, -5, 3)
        # ylims!(ax, 1e-2, 1e0)
        #
        xlims!(ax, -6, 4)
        ylims!(ax, 1e-3, 3e0)
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
    file = joinpath(plotdir, "ns-dissipation-$(n_les).pdf")
    @info "Saving dissipation plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    fig
end
