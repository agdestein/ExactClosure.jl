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
using Random
using StructuralClosure: NavierStokes
using Turbulox
using WGLMakie
using WriteVTK

@info "Loading case"
flush(stderr)

# begin
#     case = NavierStokes.smallcase()
#     n_les = 50
#     compression = 3
# end

# begin
#     case = NavierStokes.largecase()
#     n_les = 100
#     compression = 5
# end

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

# experiment = "volavg"
# sfs = NavierStokes.sfs_tensors_volume(;
#     ustart,
#     g_dns,
#     g_les,
#     poisson_dns,
#     poisson_les,
#     viscosity,
#     compression,
#     doproject = false,
# );

# experiment = "project_volavg"
# sfs = NavierStokes.sfs_tensors_volume(;
#     ustart,
#     g_dns,
#     g_les,
#     poisson_dns,
#     poisson_les,
#     viscosity,
#     compression,
#     doproject = true,
# );

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

false && let
    x, y, z = X(), Y(), Z()
    i, j = x, x
    sfs.classic[i, j].data[1, :, :] |> Array |> heatmap
    sfs.swapfil[i, j].data[1, :, :] |> Array |> heatmap
    # sfs.classic[i, j].data[:, :, 1]
    # sfs.swapfil[i, j].data[:, :, 1]
    # norm(1.0*sfs.classic.data - sfs.swapfil.data) / norm(sfs.swapfil.data)
end

true && let
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
    Turbulox.volumefilter!(fu, u, compression)
    d_classic = ScalarField(g_les)
    d_swapfil = ScalarField(g_les)
    d_swapfil_symm = ScalarField(g_les)
    d_rfu = ScalarField(g_les)
    apply!(Turbulox.tensordissipation_staggered!, g_les, d_classic, sfs.σ_classic, fu)
    apply!(Turbulox.tensordissipation_staggered!, g_les, d_swapfil, sfs.σ_swapfil, fu)
    apply!(
        Turbulox.tensordissipation_staggered!,
        g_les,
        d_swapfil_symm,
        sfs.σ_swapfil_symm,
        fu,
    )
    # apply!(Turbulox.tensordissipation_staggered!, g_les, d_rfu, sfs.rfu, fu)
    fig = Figure(; size = (400, 300))
    # ax = Axis(fig[1, 1]; xticks = ([1, 2], ["Classic", "Filter-swap"]))
    ax = Axis(fig[1, 1]; xticks = (1:4, ["No-model", "Classic", "Swap-sym", "Swap"]))
    # ax = Axis(fig[1, 1]; xticks = ([1, 2, 3], ["Classic", "Filter-swap", "Stress"]))
    boxplot!(
        ax,
        fill(1, 1),
        zeros(1);
        show_outliers = false,
        whiskerwidth = 0.2,
        orientation = :vertical,
        # label = "Classic",
    )
    boxplot!(
        ax,
        fill(2, length(d_classic)),
        d_classic.data |> vec |> Array;
        show_outliers = false,
        whiskerwidth = 0.2,
        orientation = :vertical,
        # label = "Classic",
    )
    boxplot!(
        ax,
        fill(3, length(d_swapfil_symm)),
        d_swapfil_symm.data |> vec |> Array;
        show_outliers = false,
        whiskerwidth = 0.2,
        orientation = :vertical,
        # label = "Filter-swap",
    )
    boxplot!(
        ax,
        fill(4, length(d_swapfil)),
        d_swapfil.data |> vec |> Array;
        show_outliers = false,
        whiskerwidth = 0.2,
        orientation = :vertical,
        # label = "Filter-swap",
    )
    # boxplot!(
    #     ax,
    #     fill(3, length(d_rfu)),
    #     d_rfu.data |> vec |> Array;
    #     show_outliers = false,
    #     whiskerwidth = 0.2,
    #     orientation = :vertical,
    #     # label = "Filter-swap",
    # )
    file = joinpath(plotdir, "$(experiment)-ns-dissipation.pdf")
    @info "Saving dissipation plot to $file"
    flush(stderr)
    save(file, fig; backend = CairoMakie)
    # ylims!(ax, -0.0005, 0.0005)
    fig
    # @show std(d_classic.data) std(d_swapfil.data)
    # d_swapfil.data |> vec |> Array;
end
