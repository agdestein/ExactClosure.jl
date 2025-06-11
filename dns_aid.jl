if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

using Adapt
using CairoMakie
using CUDA
using JLD2
using LinearAlgebra
using Printf
using Random
using StructuralClosure: NavierStokes
using Turbulox
using WGLMakie

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

(; viscosity, outdir, datadir, plotdir, seed) = case
g_dns = case.grid
g_les = Grid(; g_dns.ho, g_dns.backend, g_dns.L, n = n_les)
T = typeof(g_dns.L)
@assert n_les * compression == g_dns.n

poisson_dns = poissonsolver(g_dns);
poisson_les = poissonsolver(g_les);

ustart = let
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    VectorField(g_dns, data)
end

# experiment = "volume-own-pressure"
# sols, relerr = NavierStokes.dns_aid(;
#     ustart,
#     g_dns,
#     g_les,
#     poisson_dns,
#     poisson_les,
#     viscosity,
#     compression,
# );

# experiment = "volume-dns-pressure"
# sols, relerr = NavierStokes.dns_aid_pressure(;
#     ustart,
#     g_dns,
#     g_les,
#     poisson_dns,
#     poisson_les,
#     viscosity,
#     compression,
# );

experiment = "volavg"
sols, relerr = NavierStokes.dns_aid_volavg(;
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
sols, relerr = NavierStokes.dns_aid_volavg(;
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
sols, relerr = NavierStokes.dns_aid_surface(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject = true,
);

plotdir = "~/Projects/StructuralErrorPaper/figures/$experiment" |> expanduser |> mkpath

(;
    nomodel = norm(sols.nomodel.data - sols.dns_fil.data) / norm(sols.dns_fil.data),
    classic = norm(sols.classic.data - sols.dns_fil.data) / norm(sols.dns_fil.data),
    swapfil_symm = norm(sols.swapfil_symm.data - sols.dns_fil.data) /
                   norm(sols.dns_fil.data),
    swapfil = norm(sols.swapfil.data - sols.dns_fil.data) / norm(sols.dns_fil.data),
) |> pairs

relerr |> pairs

relerr.time
relerr.classic

let
    println(join(string.(keys(relerr)), " "))
    join(map(relerr) do v
        @sprintf "\$%.3g\$" v[1]
    end, " & ") |> println
    join(map(relerr) do v
        @sprintf "\$%.3g\$" v[end]
    end, " & ") |> println
end

# relerr = NavierStokes.dns_aid_surface(;
#     rhs!,
#     ustart,
#     g_dns,
#     k_les,
#     solver_dns!,
#     solver_les!,
#     viscosity,
#     compression,
# )

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Time")
    lines!(ax, relerr.time, relerr.nomodel; label = "No-model")
    lines!(ax, relerr.time, relerr.classic; label = "Classic")
    lines!(ax, relerr.time, relerr.swapfil_symm; label = "Swap-sym")
    lines!(ax, relerr.time, relerr.swapfil; label = "Swap")
    # axislegend(ax; position = :lt, framevisible = true)
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        orientation = :horizontal,
        # nbanks = 2,
        framevisible = false,
    )
    rowgap!(fig.layout, 5)
    # ylims!(ax, -0.03, 0.34)
    save("$(plotdir)/ns-error.pdf", fig; backend = CairoMakie, size = (400, 330))
    @info "Saving to $plotdir/ns-error.pdf"
    fig
end

relerr.nomodel[end]
relerr.classic[end]
relerr.swapfil[end]
relerr.swapfil_symm[end]

"Plot zoom-in box."
function zoom!(subfig, ax1; point, logx, logy, relwidth, relheight)
    # sk, se = sqrt(logx), sqrt(logy)
    sk, se = logx, logy
    kk, ee = point
    k0, k1 = kk / sk, kk * sk
    e0, e1 = ee / se, ee * se
    limits = (k0, k1, e0, e1)
    lines!(
        ax1,
        [
            Point2f(k0, e0),
            Point2f(k1, e0),
            Point2f(k1, e1),
            Point2f(k0, e1),
            Point2f(k0, e0),
        ];
        color = :black,
        linewidth = 1.5,
    )
    ax2 = Axis(
        subfig;
        width = Relative(relwidth),
        height = Relative(relheight),
        halign = 0.05,
        valign = 0.10,
        limits,
        xscale = log10,
        yscale = log10,
        xticksvisible = false,
        xticklabelsvisible = false,
        xgridvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        ygridvisible = false,
        backgroundcolor = :white,
    )
    # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
    translate!(ax2.scene, 0, 0, 10)
    translate!(ax2.elements[:background], 0, 0, 9)
    ax2
end

# Plot spectrum
let
    diss = ScalarField(g_dns)
    apply!(Turbulox.dissipation!, g_dns, diss, sols.dns_ref, viscosity)
    D = sum(diss.data) / length(diss)
    diss = nothing # free up memory
    specs = map(sols) do u
        stuff = spectral_stuff(u.grid)
        spectrum(u; stuff)
    end
    xslope = specs.dns_ref.k[8:(end-12)]
    yslope = @. 1.58 * D^(2 / 3) * (2π * xslope)^(-5 / 3)
    specs = (; specs..., kolmogo = (; k = xslope, s = yslope))
    fig = Figure()
    ax_full = Makie.Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "Wavenumber")
    o = 10
    ax_zoom = zoom!(
        fig[1, 1],
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
    Legend(
        fig[0, 1],
        ax_full;
        tellwidth = false,
        orientation = :horizontal,
        nbanks = 3,
        framevisible = false,
    )
    # vlines!(ax_full, [g_dns.n / 2, g_les.n / 2])
    # vlines!(ax_full, 150)
    # ylims!(ax_full, 1e-4, 1e-1)
    rowgap!(fig.layout, 5)
    save("$plotdir/ns-spectra.pdf", fig; backend = CairoMakie, size = (400, 380))
    @info "Saving to $plotdir/ns-spectra.pdf"
    fig
end
