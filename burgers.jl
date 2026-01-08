# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

using CairoMakie
using ExactClosure: Burgers as B
using JLD2
using Random
# using WGLMakie
using GLMakie

setup = B.getsetup()
setup |> pairs

# Plot some initial conditions
let
    (; L, nh, kpeak, initialenergy, visc, tstop, plotdir) = setup
    g = B.Grid(L, nh)
    # rng = Xoshiro(0)
    rng = Random.default_rng()
    ustart = B.randomfield(rng, g, kpeak, initialenergy)
    uh = copy(ustart)
    sh = zero(ustart)
    xh = points_stag(g)
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u")
    lines!(ax, xh, ustart |> Array; label = "Initial")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        orientation = :horizontal,
        framevisible = false,
    )
    rowgap!(fig.layout, 5)
    save("$plotdir/initial.pdf", fig; backend = CairoMakie)
    fig
end

# Plot Gaussian kernel only
let
    (; L, plotdir) = setup
    fig = Figure(; size = (400, 800))
    nh = 8000
    Δ_ratios = [1, 2, 3]
    for (iratio, Δ_ratio) in enumerate(Δ_ratios)
        gh = B.Grid(L, nh)
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
    (; L, plotdir) = setup
    fig = Figure(; size = (800, 400))
    nh = 8000
    Δ_ratios = [1, 2, 3]
    for (iratio, Δ_ratio) in enumerate(Δ_ratios)
        axes = map([1, 2]) do itype
            gh = B.Grid(L, nh)
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
            lines!(ax, KK, gg * H / h; label = "LES filter")
            lines!(ax, KK, ff * H / h; label = "FVM filter")
            lines!(ax, KK, d * H / h; label = "LES-FVM filter")
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

# Plot Burgers solution
let
    (; L, nh, kpeak, initialenergy, visc, tstop, plotdir) = setup
    g = B.Grid(L, nh)
    rng = Xoshiro(1)
    ustart = B.randomfield(rng, g, kpeak, initialenergy)
    uh = copy(ustart)
    sh = zero(ustart)
    t = 0.0
    while t < tstop
        dt = 0.3 * B.cfl(g, uh, visc) # Propose timestep
        dt = min(dt, tstop - t) # Don't overstep
        B.timestep!(g, uh, sh, visc, dt) # Perform timestep
        t += dt
    end
    xh = B.points_stag(g)
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
B.run_dns_aided_les(setup)

series = B.load_dns_aided_les(setup);

# Compute relative errors
errseries = B.compute_errors(series);
errseries |> pairs

B.write_error_table(errseries, setup)

# Compute spectra
specseries = B.compute_spectra(series, setup);

B.plot_spectra(specseries, setup)

# Compute dissipation coefficients
diss = B.compute_dissipation(series, setup);

B.plot_dissipation(diss, setup)
