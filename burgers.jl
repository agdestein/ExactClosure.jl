# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

using CairoMakie
using CUDA
using ExactClosure
using ExactClosure.Burgers
using FFTW
using JLD2
using KernelDensity
using LinearAlgebra
using Random
using Statistics
# using WGLMakie
using GLMakie

# outdir = "~/Projects/StructuralErrorPaper" |> expanduser
outdir = joinpath(@__DIR__, "output", "Burgers") |> mkpath
plotdir = "$outdir/figures" |> mkpath

setup = let
    L = 2π
    nh = 100 * 3^3 * 5
    nH = 100 .* 3 .^ (1:3)
    Δ_ratio = 2
    visc = 5e-4
    kpeak = 10
    initialenergy = 2.0
    tstop = 0.1
    nsample = 1000
    backend = CUDA.functional() ? CUDABackend() : Burgers.AK.KernelAbstractions.CPU()
    (; L, nh, nH, Δ_ratio, visc, kpeak, initialenergy, tstop, nsample, backend)
end
setup |> pairs

# Plot Gaussian kernel only
let
    (; L) = setup
    fig = Figure(; size = (400, 800))
    nh = 8000
    Δ_ratios = [1, 2, 3]
    for (iratio, Δ_ratio) in enumerate(Δ_ratios)
        gh = Grid(L, nh)
        h = L / nh
        H = 150 * h
        Δ = Δ_ratio * H
        I = round(Int, H / 2h)
        f = fill(1 / (2I + 1), 2I + 1)
        J, g = gaussian_weights(gh, Δ; nσ = 3.3)
        K = I + J
        d = map((-K):K) do k
            sum((-J):J) do j
                i = k - j
                if abs(i) ≤ I
                    f[I+1+i] * g[J+1+j]
                else
                    zero(eltype(f))
                end
            end
        end
        II = ((-I):I) * h / H
        JJ = ((-J):J) * h / H
        KK = ((-K):K) * h / H
        ff = [zeros(K - I); f; zeros(K - I)] # Pad with zeros
        gg = [zeros(K - J); g; zeros(K - J)] # Pad with zeros
        islast = iratio == length(Δ_ratios)
        ax = Axis(
            fig[iratio, 1];
            xlabel = "x / h",
            ylabel = "Weight",
            # xticksvisible = islast,
            # xticklabelsvisible = islast,
            xlabelvisible = islast,
        )
        # xlims!(ax, -3, 3)
        # ylims!(ax, 1e-5, 1)
        Label(
            fig[iratio, 1],
            "Δ = $(Δ_ratio)h";
            # fontsize = 26,
            font = :bold,
            padding = (10, 10, 10, 10),
            halign = :right,
            valign = :top,
            tellwidth = false,
            tellheight = false,
            color = Makie.wong_colors()[1],
        )
        lines!(ax, KK, gg * H / h; label = "LES-filter")
        lines!(ax, KK, ff * H / h; label = "Grid-filter")
        lines!(ax, KK, d * H / h; label = "Double-filter")
        # R, w = gaussian_weights(gh, sqrt(Δ^2 + H^2); nσ = 5)
        # scatter!(ax, (-R:R) * h / H, w; marker = :circle, label = "Haha")
        iratio == 1 && Legend(
            fig[0, 1],
            ax;
            tellwidth = false,
            orientation = :horizontal,
            framevisible = false,
        )
    end
    # rowgap!(fig.layout, 10)
    save("$(plotdir)/burgers_filters_gaussian.pdf", fig; backend = CairoMakie)
    fig
end

# Plot continuous kernels
let
    (; L) = setup
    fig = Figure(; size = (800, 400))
    nh = 8000
    Δ_ratios = [1, 2, 3]
    for (iratio, Δ_ratio) in enumerate(Δ_ratios)
        axes = map([1, 2]) do itype
            gh = Grid(L, nh)
            h = L / nh
            H = 150 * h
            Δ = Δ_ratio * H
            I = round(Int, H / 2h)
            f = fill(1 / (2I + 1), 2I + 1)
            if itype == 1
                J = round(Int, Δ / h / 2)
                g = fill(1 / (2J + 1), 2J + 1)
            elseif itype == 2
                J, g = gaussian_weights(gh, Δ; nσ = 3.3)
            end
            K = I + J
            d = map((-K):K) do k
                sum((-J):J) do j
                    i = k - j
                    if abs(i) ≤ I
                        f[I+1+i] * g[J+1+j]
                    else
                        zero(eltype(f))
                    end
                end
            end
            II = ((-I):I) * h / H
            JJ = ((-J):J) * h / H
            KK = ((-K):K) * h / H
            ff = [zeros(K - I); f; zeros(K - I)] # Pad with zeros
            gg = [zeros(K - J); g; zeros(K - J)] # Pad with zeros
            islast = iratio == length(Δ_ratios)
            typelast = itype == 2
            ax = Axis(
                fig[itype, iratio];
                xlabel = "x / h",
                ylabel = "Weight",
                xticksvisible = typelast,
                xticklabelsvisible = typelast,
                xlabelvisible = typelast,
                ylabelvisible = iratio == 1,
            )
            reach = 1.5 * Δ_ratio
            # xlims!(ax, -reach, reach)
            # ylims!(ax, 1e-5, 1)
            Label(
                fig[itype, iratio],
                (itype == 1 ? "Top-hat" : "Gaussian");
                # fontsize = 26,
                font = :bold,
                padding = (10, 10, 10, 10),
                halign = :left,
                valign = :top,
                tellwidth = false,
                tellheight = false,
                color = Makie.wong_colors()[1],
            )
            Label(
                fig[itype, iratio],
                "Δ = $(Δ_ratio)h";
                # fontsize = 26,
                font = :bold,
                padding = (10, 10, 10, 10),
                halign = :right,
                valign = :top,
                tellwidth = false,
                tellheight = false,
                color = Makie.wong_colors()[1],
            )
            lines!(ax, KK, gg * H / h; label = "LES-filter")
            lines!(ax, KK, ff * H / h; label = "Grid-filter")
            lines!(ax, KK, d * H / h; label = "Double-filter")
            # R, w = gaussian_weights(gh, sqrt(Δ^2 + H^2); nσ = 5)
            # scatter!(ax, (-R:R) * h / H, w; marker = :circle, label = "Haha")
            (itype, iratio) == (1, 1) && Legend(
                fig[0, 1:length(Δ_ratios)],
                ax;
                tellwidth = false,
                orientation = :horizontal,
                framevisible = false,
            )
            ax
        end
        linkxaxes!(axes...)
    end
    # rowgap!(fig.layout, 10)
    save("$(plotdir)/burgers_filters.pdf", fig; backend = CairoMakie)
    fig
end

# Check weights
let
    (; L, nh, nH, Δ_ratio) = setup
    fig = Figure(; size = (400, 800))
    n_nH = length(nH)
    for (igrid, nH) in enumerate(nH)
        gh = Grid(L, nh)
        gH = Grid(L, nH)
        h = spacing(gh)
        H = spacing(gH)
        Δ = 2 * H
        comp = div(nh, nH)
        I, f = tophat_weights(gh, comp)
        # J, g = tophat_weights(gh, 3 * comp)
        J, g = gaussian_weights(gh, Δ; nσ = 3)
        K = I + J
        d = map((-K):K) do k
            sum((-J):J) do j
                i = k - j
                if abs(i) ≤ I
                    f[I+1+i] * g[J+1+j]
                else
                    zero(eltype(f))
                end
            end
        end
        II = ((-I):I) * h / H
        JJ = ((-J):J) * h / H
        KK = ((-K):K) * h / H
        ff = [zeros(K - I); f; zeros(K - I)] # Pad with zeros
        gg = [zeros(K - J); g; zeros(K - J)] # Pad with zeros
        # # Check that double-weights are correct
        # uh = randomfield(Xoshiro(0), gh, 30, setup.a)
        # uh_bar = zeros(gh.n)
        # uH_bar = zeros(gH.n)
        # convolution!(gh, (J, g), uh_bar, uh)
        # volavg_stag!(gH, gh, uH_bar, uh_bar)
        # convolution!(gh, (K, d), uh_bar, uh)
        # uh_bar_H = uh_bar[comp:comp:end]
        # @show norm(uh_bar_H - uH_bar) / norm(uH_bar)
        islast = igrid == n_nH
        ax = Axis(
            fig[igrid, 1];
            xlabel = "x / h",
            ylabel = "Weight",
            xticksvisible = islast,
            xticklabelsvisible = islast,
            xlabelvisible = islast,
            # yscale = log10, 
        )
        xlims!(ax, -2.6, 2.6)
        # ylims!(ax, 1e-5, 1)
        Label(
            fig[igrid, 1],
            "N = $nH";
            # fontsize = 26,
            font = :bold,
            padding = (10, 10, 10, 10),
            halign = :right,
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
        scatter!(ax, KK, gg; marker = :rect, label = "LES-filter")
        scatter!(ax, KK, ff; marker = :diamond, label = "Grid-filter")
        scatter!(ax, KK, d; marker = :circle, label = "Double-filter")
        # R, w = gaussian_weights(gh, sqrt(Δ^2 + H^2); nσ = 5)
        # scatter!(ax, (-R:R) * h / H, w; marker = :circle, label = "Haha")
        islast && Legend(
            fig[0, 1],
            ax;
            tellwidth = false,
            orientation = :horizontal,
            framevisible = false,
        )
    end
    # rowgap!(fig.layout, 10)
    save("$(plotdir)/burgers_filters_discrete.pdf", fig; backend = CairoMakie)
    fig
end

using CUDA

# Plot Burgers solution
let
    (; L, nh, kpeak, initialenergy, visc, tstop) = setup
    g = Grid(L, nh)
    ustart = randomfield(Xoshiro(0), g, kpeak, initialenergy)
    uh = copy(ustart)
    sh = zero(ustart)
    t = 0.0
    while t < tstop
        dt = 0.3 * cfl(g, uh, visc) # Propose timestep
        dt = min(dt, tstop - t) # Don't overstep
        timestep!(g, uh, sh, visc, dt) # Perform timestep
        t += dt
    end
    xh = points_stag(g)
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u")
    lines!(ax, xh, ustart |> Array; label = "Initial")
    lines!(ax, xh, uh |> Array; label = "Final")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        orientation = :horizontal,
        framevisible = false,
    )
    rowgap!(fig.layout, 5)
    save("$plotdir/burgers_solution.pdf", fig; backend = CairoMakie)
    # @show sum(abs2, ustart) / 2 / nh sum(abs2, uh) / 2 / nh
    fig
end

# DNS-aided LES
series = map(setup.nH) do nH
    (; L, nh, Δ_ratio, visc, kpeak, initialenergy, tstop, nsample, backend) = setup
    # backend = Burgers.AK.KernelAbstractions.CPU()
    gh = Grid(L, nh)
    gH = Grid(L, nH)
    Δ = spacing(gH) * Δ_ratio
    fields = (;
        dns_ref = (; g = gh, u = zeros(nh, nsample), label = "DNS"),
        dns_fil = (; g = gH, u = zeros(nH, nsample), label = "Filtered DNS"),
        nomodel = (; g = gH, u = zeros(nH, nsample), label = "No model"),
        class_m = (; g = gH, u = zeros(nH, nsample), label = "Classic"),
        class_p = (; g = gH, u = zeros(nH, nsample), label = "Classic+"),
        swapfil = (; g = gH, u = zeros(nH, nsample), label = "Swap (ours)"),
    )
    # Threads.@threads for i = 1:nsample
    for i = 1:nsample
        @info "N = $nH, sample $i of $nsample"
        rng = Xoshiro(i)
        ustart = randomfield(rng, gh, kpeak, initialenergy) |> adapt(backend)
        sols = Burgers.dns_aided_les(
            ustart,
            gh,
            gH,
            visc;
            Δ,
            tstop,
            cfl_factor = 0.3,
            lesfiltertype = :gaussian,
        )
        for key in keys(fields)
            copyto!(view(fields[key].u, :, i), sols[key])
        end
    end
    (; nH, fields)
end;

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
    linkaxes!(axes...)
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
        xlims!(ax, -0.3 * a, 0.5 * a)
        ylims!(ax, 1e-4, 4e-1)
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
