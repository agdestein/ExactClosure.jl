using ExactClosure: NavierStokes as NS
using GLMakie
using Random

u = let
    n = 300
    visc = 3e-4
    l = 1.0
    grid = NS.Grid{2}(l, n)
    backend = NS.KernelAbstractions.CPU()
    poisson = NS.poissonsolver(grid, backend)
    p = NS.Field(grid, backend)
    # u = NS.Field(grid, backend), NS.Field(grid, backend)
    profile, args = k -> (k > 0) * k^-3.0, (;)
    # profile, args = NS.peak_profile, (; kpeak = 5)
    u = NS.randomfield(
        profile,
        grid,
        backend,
        poisson;
        rng = Xoshiro(0),
        totalenergy = 1.0,
        args...,
    )
    NS.divergence!(p, u)
    w = NS.Field(grid, backend)
    du = NS.Field(grid, backend), NS.Field(grid, backend)
    u0 = NS.Field(grid, backend), NS.Field(grid, backend)
    wobs = Makie.Observable(w.data)
    fig = heatmap(wobs)
    fig |> display
    i = 0
    t = 0.0
    tmax = 5.0
    while t < tmax
        if i > 0 # Skip first step
            dt = NS.propose_timestep(u, visc, 0.5)
            dt = min(dt, tmax - t) # Don't step past tmax
            # NS.step_forwardeuler!(u, du, p, poisson, visc, dt)
            NS.step_wray3!(u, du, u0, p, poisson, visc, dt)
            t += dt
        end
        if i % 1 == 0
            @show t
            NS.vorticity!(w, u)
            wobs[] = w.data
            # display(fig)
        end
        i += 1
    end
    u
end

let
    grid = u[1].grid
    backend = NS.KernelAbstractions.CPU()
    v = NS.vectorfield(grid, backend)
    for i in eachindex(u)
        ui, vi = u[i], v[i]
        NS.AK.foreachindex(ui.data) do ilin
            I = NS.linear2cartesian(grid, ilin)
            vi[I] = (ui[I] + ui[NS.right(I, i, -1)]) / 2
        end
    end
    poisson = NS.poissonsolver(grid, backend)
    stuff = NS.spectral_stuff(grid, backend)
    specu = NS.spectrum(u, stuff, poisson)
    specv = NS.spectrum(v, stuff, poisson)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "k", ylabel = "E(k)", xscale = log10, yscale = log10)
    lines!(ax, specu.k, specu.s)
    lines!(ax, specv.k, specv.s)
    kkol = [10^1.5, maximum(specu.k)]
    ekol = 1e3 * kkol .^ (-3)
    lines!(ax, kkol, ekol)
    fig
end

let
    n = 500
    # n = size(u[1], 1)
    dt = 1e-4
    visc = 3e-4
    l = 1.0
    grid = NS.Grid{2}(l, n)
    h = NS.spacing(grid)
    backend = NS.KernelAbstractions.CPU()
    comp = 25
    H = comp * h
    k_th = NS.tophat(grid, comp)
    k_gauss = NS.gaussian(grid, 2H, 3)
    ki = NS.composekernel(k_th, k_gauss)
    # ki = NS.tophat(grid, comp)
    # ki = NS.gaussian(grid, 2H, 3)
    kernel = NS.kernelproduct(grid, ntuple(Returns(ki), NS.dim(grid)))
    # let
    #     fig = Figure()
    #     ax = Axis(fig[1, 1]; xlabel = "x / h", ylabel = "Weight")
    #     lines!(ax, k_th[1] ./ h, k_th[2], label = "Tophat")
    #     lines!(ax, k_gauss[1] ./ h, k_gauss[2], label = "Gaussian")
    #     lines!(ax, ki[1] ./ h, ki[2], label = "Composition")
    #     axislegend(ax)
    #     fig
    # end
    profile, args = k -> (k > 0) * k^-3.0, (;)
    poisson = NS.poissonsolver(grid, backend)
    p = NS.Field(grid, backend)
    u = NS.randomfield(
        profile,
        grid,
        backend,
        poisson;
        rng = Xoshiro(0),
        totalenergy = 1.0,
        args...,
    )
    ubar = NS.vectorfield(grid, backend)
    @show eltype(ubar[1])
    for (ubar, u) in zip(ubar, u)
        NS.applyfilter!(ubar, u, kernel)
    end
    # @report_opt NS.applyfilter!(ubar[1], u[1], kernel)
    grid = u[1].grid
    stuff = NS.spectral_stuff(grid, backend)
    spec = NS.spectrum(u, stuff, poisson)
    specbar = NS.spectrum(ubar, stuff, poisson)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "k", ylabel = "E(k)", xscale = log10, yscale = log10)
    lines!(ax, spec.k, spec.s)
    lines!(ax, specbar.k, specbar.s)
    kkol = [10^1.5, maximum(spec.k)]
    ekol = 1e4 * kkol .^ (-3)
    lines!(ax, kkol, ekol)
    fig
end

uaid = NS.dnsaid()

uaid.u_dns[1] |> heatmap
uaid.u_nomo[1] |> heatmap
uaid.u_cfd[1] |> heatmap

uaid = NS.dnsaid()
uaid = NS.dnsaid_project()

NS.compute_errors(uaid)

NS.compute_errors(uaid) |> pairs

NS.plot_spectra(uaid)

