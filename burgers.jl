# localhost:9384/browser-display

# Run with julia -t auto to speed up simulations

if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

using CairoMakie
using FFTW
using LinearAlgebra
using Random
using StructuralClosure: Burgers
using WGLMakie

outdir = "~/Projects/StructuralErrorPaper" |> expanduser
plotdir = "$outdir/figures"

setup = let
    L = 2π
    nh = 3^8
    nH = 3 .^ (5:7)
    visc = 5e-4
    kp = 10
    A = 2 / kp / 3 / sqrt(π)
    a = sqrt(2 * A)
    tstop = 0.1
    nsample = 1000
    seed = 1234
    (; L, nh, nH, visc, kp, a, tstop, nsample, seed)
end
setup |> pairs

# Show stepping
let
    (; L, nh, kp, a, visc, tstop, seed) = setup
    g = Grid(L, nh)
    Random.seed!(seed)
    ustart = randomfield(g, kp, a)
    uh = copy(ustart)
    t = 0.0
    while t < tstop
        dt = 0.3 * cfl(g, uh, visc)
        dt = min(dt, tstop - t) # Don't overstep
        uh = timestep(g, uh, visc, dt)
        t += dt
    end
    xh = points_stag(g)
    @show sum(abs2, ustart) / 2nh sum(abs2, uh) / 2nh
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "x")
    lines!(ax, xh, ustart; label = "Initial")
    lines!(ax, xh, uh; label = "Final")
    # lines!(ax, xH, uH)
    # axislegend(ax; position = :lb)
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        orientation = :horizontal,
        # nbanks = 2,
        framevisible = false,
    )
    rowgap!(fig.layout, 5)
    save("$plotdir/burgers_solution.pdf", fig; backend = CairoMakie)
    fig
end

series = map(setup.nH) do nH
    @show nH
    (; L, nh, visc, kp, a, tstop, nsample, seed) = setup
    gh = Grid(L, nh)
    gH = Grid(L, nH)
    fields = (;
        dns_ref = (; g = gh, u = zeros(nh, nsample), label = "DNS"),
        dns_fil = (; g = gH, u = zeros(nH, nsample), label = "Filtered DNS"),
        nomodel = (; g = gH, u = zeros(nH, nsample), label = "No model"),
        classic = (; g = gH, u = zeros(nH, nsample), label = "Classic"),
        revolut = (; g = gH, u = zeros(nH, nsample), label = "Swap"),
        # classic_conv = (; g = gH, u = zeros(nH, nsample), label = "Classic (conv)"),
        # revolut_conv = (; g = gH, u = zeros(nH, nsample), label = "Filter-swap (conv)"),
    )
    Random.seed!(seed)
    Threads.@threads for i = 1:nsample
        ustart = randomfield(gh, kp, a)
        sols = Burgers.dns_aided_les(ustart, gh, gH, visc; tstop, cfl_factor = 0.3)
        for key in keys(fields)
            copyto!(view(fields[key].u, :, i), sols[key])
        end
    end
    (; nH, fields)
end;

errseries = let
    fields = map([:nomodel, :classic, :revolut]) do key
        e = map(series) do (; nH, fields)
            (; u) = fields[key]
            norm(u - fields.dns_fil.u) / norm(fields.dns_fil.u)
        end
        key => (; e, series[1].fields[key].label)
    end
    (; nH = getindex.(series, :nH), fields = (; fields...))
end;
errseries.fields |> pairs

open("$outdir/tables/burgers_error.tex", "w") do io
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

let
    fig = Figure(; size = (400, 300))
    ax = Axis(
        fig[1, 1];
        xlabel = "Number of grid points",
        xscale = log10,
        yscale = log10,
        xticks = errseries.nH,
    )
    for (; e, label) in errseries.fields
        scatterlines!(
            ax,
            errseries.nH,
            e;
            label,
            # marker = :circle,
        )
    end
    axislegend(ax; position = :cc)
    # save("$(plotdir)/burgers_error.pdf", fig; backend = CairoMakie)
    fig
end

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

# Plot zoom-in box
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
        valign = 0.05,
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

# Spectrum
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
            # xscale = log2,
            xscale = log10,
            # xticks = ticks,
            xlabel = "Wavenumber",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
            # ylabel = "N = $nH",
        )
        tip = specs.dns_fil
        o = (22, 70, 210)[i]
        ax_zoom = zoom!(
            f[i, 1],
            ax;
            point = (tip.k[end-o], tip.e[end-o]),
            logx = 1.3,
            logy = 1.6,
            relwidth = 0.45,
            relheight = 0.45,
        )
        styles = (;
            dns_ref = (; color = Cycled(1)),
            dns_fil = (; color = Cycled(1), linestyle = :dash),
            nomodel = (; color = Cycled(2)),
            classic = (; color = Cycled(3)),
            revolut = (; color = Cycled(4)),
        )
        for key in [
            :dns_ref,
            :nomodel,
            :classic,
            :revolut,
            # :classic_conv,
            # :revolut_conv,
            :dns_fil,
        ]
            (; k, e, label) = specs[key]
            lines!(ax, k, e; label, styles[key]...)
            lines!(ax_zoom, k, e; styles[key]...)
            # lines!(ax, k[1:(end-1)], specavg3(e)[1:(end-1)]; label)
            # lines!(ax, k[1:end-2], specavg5(e)[1:end-2]; label)
        end
        # theoretical = let
        #     L = 2π
        #     (; k, e) = specs.dns_ref
        #     n = nH
        #     G = @. sin(k * L / 2n) * 2n / k / L
        #     Ge = @. G^2 * e
        #     k, Ge
        # end
        # lines!(ax, theoretical...)
        # lines!(ax, specs[1].k[29:300], k -> 1.5e0 * k^(-2); label = "k^-2")
        # vlines!(ax, div(nH, 2); color = :black)
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

let
    r = stress(gh, sols.uh, visc)
    d = dissipation(gh, sols.uh, r)
    hist(d; bins = 50)
end

diss = let
    (; visc) = setup
    map(series) do (; nH, fields)
        gh = Grid(setup.L, setup.nh)
        gH = Grid(setup.L, nH)
        d = stack(eachcol(fields.dns_ref.u)) do uh
            su = map(i -> stress(gh, uh, visc, i), 1:gh.n)
            vsu = volavg_coll!(gH, gh, zeros(nH), su)
            fsu = suravg_coll!(gH, gh, zeros(nH), su)
            vu = volavg_stag!(gH, gh, zeros(nH), uh)
            svu = map(i -> stress(gH, vu, visc, i), 1:gH.n)
            σ_classic = vsu - svu
            σ_revolut = fsu - svu
            d_classic = dissipation(gH, vu, σ_classic)
            d_revolut = dissipation(gH, vu, σ_revolut)
            hcat(d_classic, d_revolut)
        end
        (; nomodel = zeros(1), classic = d[:, 1, :] |> vec, revolut = d[:, 2, :] |> vec)
    end
end;

# Multiple trajectories
let
    models = [
        (; label = "No-model", sym = :nomodel),
        (; label = "Classic", sym = :classic),
        (; label = "Swap", sym = :revolut),
    ]
    fig = Figure(; size = (400, 800))
    for (i, diss) in diss |> enumerate
        ax = Axis(
            fig[i, 1];
            xticks = (eachindex(models), getindex.(models, :label)),
            xticksvisible = i == 3,
            xticklabelsvisible = i == 3,
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
            fig[i, 1], #, TopLeft()],
            "N = $(setup.nH[i])";
            # fontsize = 26,
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
