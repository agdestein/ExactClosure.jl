# DNS-aided LES for the Burgers equation
#
# Run with `julia -t auto` to run the simulations in parallel.

# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

using CairoMakie
using ExactClosure
using ExactClosure: Burgers as B
using FFTW
using JLD2
using KernelDensity
using LinearAlgebra
using Random
# using WGLMakie
using GLMakie

setup = B.getsetup()
setup |> pairs

dnsdata = B.create_dns(setup; cfl_factor = 0.3);

# save_object("$(setup.outdir)/burgers_dns.jld2", dnsdata)
# dnsdata = load_object("$(setup.outdir)/burgers_dns.jld2")

fields = B.smagorinsky_fields(setup, dnsdata; lesfiltertype = :gaussian);

# Estimate θ
θ_classic = map(fields) do (; Sh, Th, D_Sh, D_Th)
    θ2 = -dot(Sh, Th) / dot(Sh, Sh)
    # θ2 = -dot(D_Sh, D_Th) / dot(D_Sh, D_Sh)
    # θ2 = -sum(D_Th) / sum(D_Sh)
    sqrt(θ2)
end

θ_discret = map(fields) do (; SH, TH, D_SH, D_TH)
    θ2 = -dot(SH, TH) / dot(SH, SH)
    # θ2 = -dot(D_SH, D_TH) / dot(D_SH, D_SH)
    # θ2 = -sum(D_TH) / sum(D_SH)
    sqrt(θ2)
end

u_classic = B.solve_smagorinsky(setup, dnsdata, θ_classic)
u_discret = B.solve_smagorinsky(setup, dnsdata, θ_discret)

# save_object("$(setup.outdir)/burgers_classic.jld2", u_classic)
# save_object("$(setup.outdir)/burgers_discret.jld2", u_discret)

# u_classic = load_object("$(setup.outdir)/burgers_classic.jld2")
# u_discret = load_object("$(setup.outdir)/burgers_discret.jld2")

let
    j = 1
    fig = Figure(; size = (400, 800))
    for (i, nH) in enumerate(setup.nH)
        gh = B.Grid(setup.L, setup.nh)
        gH = B.Grid(setup.L, nH)
        xh = B.points_stag(gh)
        xH = B.points_stag(gH)
        ax = Axis(fig[i, 1]; title = "N = $nH", xlabel = "x", ylabel = "u")
        lines!(ax, xh, dnsdata[2][:, j]; label = "DNS")
        # lines!(ax, xH, u_classic[i][:, j]; label = "Smagorinsky")
        lines!(ax, xH, u_discret[i][:, j]; label = "Smagorinsky")
    end
    linkaxes!(fig.content...)
    fig
end

# Plot SFS for different resolutions
let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x")
    for i = 1:1
        (; Th, TH) = fields[i]
        j = 1
        # t = Th[:, j]
        t = TH[:, j]
        lines!(ax, t)
    end
    fig
end

# Plot solution, Smagorinsky SFS, and true SFS
let
    j = 1
    (; Sh, Th, SH, TH, Ubar) = fields[j]
    Ustart, U = dnsdata
    i = 1
    ustart = Ustart[:, i]
    u = U[:, i]
    sh = Sh[:, i]
    th = Th[:, i]
    sH = SH[:, i]
    tH = TH[:, i]
    g = Grid(setup.L, setup.nh)
    xs = points_stag(g)
    xc = points_coll(g)
    H = setup.L / setup.nH[j]
    Δ = setup.Δ_ratio * H
    fac = Δ^2 + H^2
    fig = Figure()
    axu = Axis(fig[1, 1]; ylabel = "Velocity", xticklabelsvisible = false)
    lines!(axu, xs, ustart; color = Cycled(2), label = "Initial DNS")
    lines!(axu, xs, u; color = Cycled(1), label = "Final DNS")
    scatterlines!(
        axu,
        points_stag(Grid(setup.L, setup.nH[j])),
        # u_classic[j][:, i];
        u_discret[j][:, i];
        color = Cycled(3),
        label = "Smag",
    )
    lines!(axu, xs, Ubar[:, i]; color = Cycled(3), label = "Filtered")
    axislegend(axu)
    #
    axth = Axis(fig[2, 1]; xticklabelsvisible = false, ylabel = "Flux")
    lines!(axth, xc, th; label = "Classic SFS")
    lines!(axth, xc, -θ_classic[j]^2 * sh; label = "Classic Smagorinsky")
    axislegend(axth; position = :rb)
    #
    axtH = Axis(fig[3, 1]; xlabel = "x", ylabel = "Flux")
    lines!(axtH, xc, tH; label = "Swap SFS")
    # lines!(axtH, xc, -θ_classic[j]^2 * sh; label = "Classic Smagorinsky")
    lines!(axtH, xc, -θ_discret[j]^2 * sH; label = "Discrete Smagorinsky")
    axislegend(axtH; position = :rb)
    #
    linkxaxes!(axu, axth, axtH)
    linkyaxes!(axth, axtH)
    if false
        save("$(plotdir)/burgers_smagorinsky.pdf", fig; backend = CairoMakie)
    else
        # xlims!(axtH, 2.0, 2.4)
        # xlims!(axtH, 2.7, 3.2)
        xlims!(axtH, 3.51, 3.7)
        RR = 20 * (Δ^2 + H^2)
        # ylims!(axtH, -RR, RR / 5)
        save("$(plotdir)/burgers_smagorinsky_fit_zoom.pdf", fig; backend = CairoMakie)
    end
    fig
end

# Compute relative errors
relerrs = map(enumerate(setup.nH)) do (i, nH)
    gh = B.Grid(setup.L, setup.nh)
    gH = B.Grid(setup.L, nH)
    @show nH
    U = dnsdata[2]
    comp = div(gh.n, gH.n)
    Ubar = fields[i].Ubar[comp:comp:end, :] # Extract coarse-grid components
    U_classic = u_classic[i]
    U_discret = u_discret[i]
    e = map((; classic = U_classic, discret = U_discret)) do U
        norm(U - Ubar) / norm(Ubar)
    end
    (; nH, e)
end

# Plot coefficients
let
    colors = Makie.wong_colors()
    fig = Figure(; size = (400, 320))
    ax = Axis(
        fig[1, 1];
        # xlabel = "N",
        ylabel = "Smagorinsky coefficient",
        xticks = (eachindex(relerrs), map(nH -> "N = $nH", setup.nH)),
    )
    θ = hcat(θ_classic, θ_discret)[:]
    i = repeat(eachindex(setup.nH), 1, 2)[:]
    group = repeat((1:2)', length(setup.nH), 1)[:]
    barplot!(
        ax,
        i,
        θ;
        color = colors[group],
        dodge = group,
        bar_labels = :y,
        # label_position = :center,
        # flip_labels_at=(0.0, 0.0),
    )
    # ylims!(ax, relerrs[1].e.classic)
    ylims!(ax, -0.015, 0.35)
    labels = ["Classic", "Discrete (ours)"]
    elements = map(i -> PolyElement(; polycolor = colors[i]), eachindex(labels))
    title = "Smagorinsky coefficient"
    Legend(
        fig[0, :],
        elements,
        labels;
        # title;
        tellwidth = false,
        orientation = :horizontal,
        nbanks = 1,
        framevisible = false,
    )
    save("$(setup.plotdir)/burgers_smagorinsky_coefficients.pdf", fig; backend = CairoMakie)
    fig
end

# Plot errors
let
    colors = Makie.wong_colors()
    fig = Figure(; size = (400, 320))
    ax = Axis(
        fig[1, 1];
        # xlabel = "N",
        ylabel = "Relative Error",
        xticks = (eachindex(relerrs), map(nH -> "N = $nH", setup.nH)),
    )
    for i in eachindex(relerrs)
        (; nH, e) = relerrs[i]
        barplot!(
            ax,
            fill(i, 2),
            vcat(e...);
            color = colors[[1, 2]],
            dodge = [1, 2],
            bar_labels = :y,
            # label_position = :center,
            # flip_labels_at=(0.0, 0.0),
        )
    end
    # ylims!(ax, relerrs[1].e.classic)
    ylims!(ax, -0.004, 0.077)
    labels = ["Classic", "Discrete (ours)"]
    elements = map(i -> PolyElement(; polycolor = colors[i]), eachindex(labels))
    # title = "Smagorinsky coefficient"
    Legend(
        fig[0, :],
        elements,
        labels;
        # title;
        tellwidth = false,
        orientation = :horizontal,
        nbanks = 1,
        framevisible = false,
    )
    save("$(setup.plotdir)/burgers_smagorinsky_errors.pdf", fig; backend = CairoMakie)
    fig
end

# Write errors to LaTeX table
let
    path = joinpath(setup.outdir, "tables") |> mkpath
    file = joinpath(path, "burgers_smagorinsky_error.tex")
    open(file, "w") do io
        tab = "    "
        c = join(fill("r", length(relerrs[1]) + 1), " ")
        println(io, "\\begin{tabular}{$c}")
        println(io, tab, "\\toprule")
        keys = [:classic, :discret]
        labels = join(["Classic", "Discrete (ours)"], " & ")
        println(io, tab, "N & $labels \\\\")
        println(io, tab, "\\midrule")
        for (; nH, e) in relerrs
            println(
                io,
                tab,
                nH,
                " & ",
                join(map(k -> round(e[k]; sigdigits = 3), keys), " & "),
                " \\\\",
            )
        end
        println(io, tab, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io)
        println(io, "% vim: conceallevel=0 textwidth=0")
    end
    open(readlines, file) .|> println
    nothing
end

# Write Smagorinsky constants to LaTeX table
let
    path = joinpath(setup.outdir, "tables") |> mkpath
    file = joinpath(path, "burgers_smagorinsky_constants.tex")
    open(file, "w") do io
        tab = "    "
        println(io, "\\begin{tabular}{r r r}")
        println(io, tab, "\\toprule")
        # println(io, tab, "N & \$\\theta_\\text{classic}\$ & \$\\theta_\\text{swap}\$ \\\\")
        println(io, tab, "N & Classic & Discrete (ours) \\\\")
        println(io, tab, "\\midrule")
        for i in eachindex(setup.nH)
            nH = setup.nH[i]
            θc = round(θ_classic[i]; sigdigits = 3)
            θs = round(θ_discret[i]; sigdigits = 3)
            println(io, tab, "\$$nH\$ & \$$θc\$ & \$$θs\$ \\\\")
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
specseries = map(enumerate(setup.nH)) do (i, nH)
    gh = B.Grid(setup.L, setup.nh)
    gH = B.Grid(setup.L, nH)
    @show nH
    U = dnsdata[2]
    (; Ubar) = fields[i]
    U_classic = u_classic[i]
    U_discret = u_discret[i]
    specfields = (;
        dns_ref = (; u = U, label = "DNS"),
        dns_fil = (; u = Ubar, label = "Filtered DNS"),
        classic = (; u = U_classic, label = "Smagorinsky (classic)"),
        discret = (; u = U_discret, label = "Smagorinsky (discrete)"),
    )
    specs = map(specfields) do (; u, label)
        uhat = rfft(u, 1)
        n, nsample = size(u)
        e = sum(u -> abs2(u) / 2 / n^2 / nsample, uhat; dims = 2)
        e = 2 * e[2:end] # Multiply by 2 since RFFT is half
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
        xlims!(ax, 7e-1, 1e4)
        ylims!(ax, 1e-11, 1e-1)
        tip = specs.classic
        # o = (22, 70, 210)[i]
        o = (72, 200, 600)[i]
        ax_zoom = ExactClosure.zoombox!(
            f[i, 1],
            ax;
            point = (tip.k[end-o], tip.e[end-o]),
            logx = 1.5,
            logy = 2.8,
            relwidth = 0.45,
            relheight = 0.45,
        )
        styles = (;
            dns_ref = (; color = :black),
            dns_fil = (; color = Cycled(1)),
            # dns_fil = (; color = :black, linestyle = :dash),
            classic = (; color = Cycled(2)),
            discret = (; color = Cycled(3)),
        )
        for key in [:dns_ref, :dns_fil, :classic, :discret]
            (; k, e, label) = specs[key]
            # At the end of the spectrum, there are too many points for plotting.
            # Choose a logarithmically equispaced subset of points instead
            npoint = 300
            ii = round.(Int, logrange(1, length(k), npoint)) |> unique
            lines!(ax, k[ii], e[ii]; label, styles[key]...)
            lines!(ax_zoom, k, e; styles[key]...)
        end
        # kkol = logrange(2e0, 2e4, 100)
        # ekol = map(k -> 1.8e0 * k^-2, kkol)
        # lines!(ax, kkol, ekol; color = :black, linestyle = :dash)
        # lines!(ax_zoom, kkol, ekol; color = :black, linestyle = :dash)
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
    save("$(setup.plotdir)/burgers_smagorinsky_spectrum.pdf", fig; backend = CairoMakie)
    fig
end
