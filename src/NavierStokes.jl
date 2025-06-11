"3D simulation."
module NavierStokes

using JLD2
using LinearAlgebra
using Turbulox
using CUDA
using KernelAbstractions
using Makie
using StaticArrays

get_backend() = CUDA.functional() ? CUDABackend() : CPU()

"Prototype test case."
function smallcase()
    seed = 123
    # T = Float32
    T = Float64
    grid = Grid(; ho = Val(1), L = T(1), n = 150, backend = get_backend())
    viscosity = 0.3e-4 |> T
    outdir = joinpath(@__DIR__, "..", "output", "smallcase") |> mkpath
    datadir = joinpath(outdir, "data") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    amplitude = T(5e-2)
    kpeak = 5
    (; seed, grid, viscosity, outdir, datadir, plotdir, amplitude, kpeak)
end

"Large test case."
function largecase()
    seed = 123
    # T = Float32
    T = Float64
    grid = Grid(; ho = Val(1), L = T(1), n = 500, backend = get_backend())
    viscosity = 1 / 10_000 |> T
    outdir = joinpath(@__DIR__, "..", "output", "largecase") |> mkpath
    datadir = joinpath(outdir, "data") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    amplitude = T(5e-2)
    kpeak = 5
    (; seed, grid, viscosity, outdir, datadir, plotdir, amplitude, kpeak)
end

"Large test case."
function snelliuscase()
    seed = 123
    # T = Float32
    T = Float64
    grid = Grid(; ho = Val(1), L = T(1), n = 800, backend = get_backend())
    viscosity = 1 / 10_000 |> T
    outdir = joinpath(ENV["DEEPDIP"], "StructuralClosure", "snelliuscase") |> mkpath
    datadir = joinpath(outdir, "data") |> mkpath
    plotdir = joinpath(outdir, "plots") |> mkpath
    amplitude = T(5e-2)
    kpeak = 5
    (; seed, grid, viscosity, outdir, datadir, plotdir, amplitude, kpeak)
end

"Vorticity in the plane `Iz`."
@kernel function vorticity!(ω, u, Iz)
    x, y = X(), Y()
    Ix, Iy = @index(Global, NTuple)
    I = CartesianIndex(Ix, Iy, Iz)
    δyux = Turbulox.δ(u[x], y, I)
    δxuy = Turbulox.δ(u[y], x, I)
    ω[Ix, Iy] = -δyux + δxuy
end

function plotsol(u, p, viscosity; colorrange = (-50, 50))
    (; grid) = u
    T = eltype(u)
    ut_obs = Observable((; u, t = 0 |> T))
    ω = similar(u.data, u.grid.n, u.grid.n)
    ω_obs = ω |> Array |> Observable
    Et = Point2{T}[]
    spec_obs = Point2{T}[] |> Observable
    slope_obs = Point2{T}[] |> Observable
    Et_obs = Observable(Et)
    stuff = spectral_stuff(u.grid)
    I3 = 1
    on(ut_obs) do (; u, t)
        apply!(vorticity!, grid, ω, u, I3; ndrange = size(ω))
        e = sum(abs2, u.data) / 2 / grid.n^3
        ω_obs[] = copyto!(ω_obs[], ω)
        Et_obs[] = push!(Et, Point2(t, e))
        apply!(Turbulox.dissipation!, grid, p, u, viscosity)
        D = sum(p.data) / length(p)
        spec = spectrum(u; stuff)
        spec_obs[] = Point2.(spec.k, spec.s)
        xslope = spec.k[10:end]
        C_K = 1.58 |> T
        τ = 2π |> T
        yslope = @. C_K * D^T(2 / 3) * (τ * xslope)^T(-5 / 3)
        slope_obs[] = Point2.(xslope, yslope)
    end
    ut_obs[] = (; u, t = T(0))
    fig = Figure(; size = (1200, 450))

    ax1 =
        Makie.Axis(fig[1, 1]; title = "Vorticity at z = $(I3)", xlabel = "x", ylabel = "y")
    hm = image!(ax1, ω_obs; colorrange, colormap = :seaborn_icefire_gradient)
    # Colorbar(fig[1, 0], hm)

    ax2 = Makie.Axis(fig[1, 2]; title = "Total energy", xlabel = "Time")
    lines!(ax2, Et_obs)

    ax3 = Makie.Axis(
        fig[1, 3];
        xscale = log10,
        yscale = log10,
        title = "Energy spectrum",
        xlabel = "Wavenumber",
    )
    lines!(ax3, spec_obs; label = "DNS")
    lines!(ax3, slope_obs; label = "Kolmogorov")
    # vlines!(ax3, 1 / s.L / 1)
    # vlines!(ax3, 1 / s.λ / 1)
    axislegend(ax3; position = :lb)
    # ylims!(ax3, 1e-4, 1e-1)

    on(_ -> autolimits!(ax2), ut_obs)
    on(_ -> autolimits!(ax3), ut_obs)
    fig |> display
    ut_obs
end

tensordot(a, b) =
    mapreduce(+, a, b) do a, b
        dot(a, b)
    end

function cross_correlation(a, b)
    amean = sum(a) / length(a)
    bmean = sum(b) / length(b)
    adev = map(a -> a - amean, a)
    bdev = map(b -> b - bmean, b)
    tensordot(adev, bdev) / sqrt(tensordot(adev, adev) * tensordot(bdev, bdev))
end

"""
Flatten "n-vector of tensors" to 9-times-n array.
"""
@kernel function tensorextract!(grid, τ_flat, τ)
    i, x = @index(Global, NTuple)
    tensor = τ[i, x]
    τ_flat[i, 1, x] = tensor[1, 1]
    τ_flat[i, 2, x] = tensor[2, 1]
    τ_flat[i, 3, x] = tensor[3, 1]
    τ_flat[i, 4, x] = tensor[1, 2]
    τ_flat[i, 5, x] = tensor[2, 2]
    τ_flat[i, 6, x] = tensor[3, 2]
    τ_flat[i, 7, x] = tensor[1, 3]
    τ_flat[i, 8, x] = tensor[2, 3]
    τ_flat[i, 9, x] = tensor[3, 3]
end

"""
Convert 9-times-n array to "n-vector of tensors".
"""
@kernel function tensorback!(grid, τ, τ_flat)
    x = @index(Global, Linear)
    τ[x] = SMatrix{3,3,T,9}(
        τ_flat[1, x],
        τ_flat[2, x],
        τ_flat[3, x],
        τ_flat[4, x],
        τ_flat[5, x],
        τ_flat[6, x],
        τ_flat[7, x],
        τ_flat[8, x],
        τ_flat[9, x],
    )
end

"Load JLD2 file as named tuple (instead of dict)."
function namedtupleload(file)
    dict = load(file)
    k, v = keys(dict), values(dict)
    pairs = @. Symbol(k) => v
    (; pairs...)
end

function fe_step!(du, u, cache, Δt, poisson)
    (; p) = cache
    axpy!(Δt, du.data, u.data)
    project!(u, p, poisson)
end

function dns_aid(; ustart, g_dns, g_les, poisson_dns, poisson_les, viscosity, compression)
    cache_dns = (; du = VectorField(g_dns), p = ScalarField(g_dns))
    cache_les = (; du = VectorField(g_les), p = ScalarField(g_les))
    T = typeof(g_dns.L)
    udns = VectorField(g_dns, copy(ustart.data))
    t = 0 |> T
    cfl = 0.15 |> T
    tstop = 0.1 |> T
    fru = TensorField(g_les)
    firu = TensorField(g_les)
    fu = VectorField(g_les)
    Turbulox.volumefilter!(fu, udns, compression)
    unomodel = VectorField(g_les, copy(fu.data)) # Not divergence-free
    project!(unomodel, cache_les.p, poisson_les) # Make divergence free
    u = (;
        dns_ref = udns,
        dns_fil = fu,
        nomodel = unomodel,
        classic = VectorField(g_les, copy(unomodel.data)),
        swapfil = VectorField(g_les, copy(unomodel.data)),
        swapfil_symm = VectorField(g_les, copy(unomodel.data)),
    )
    relerr = (;
        time = zeros(0),
        nomodel = zeros(0),
        classic = zeros(0),
        swapfil = zeros(0),
        swapfil_symm = zeros(0),
    )
    i = 0
    while t < tstop
        # Skip first iter to get initial errors
        if i > 0
            ru = stresstensor(u.dns_ref, viscosity)
            rfu = stresstensor(u.dns_fil, viscosity)
            Turbulox.volumefilter!(fru, ru, compression)
            Turbulox.surfacefilter!(firu, ru, compression, false)

            # DNS
            Δt = cfl * propose_timestep(u.dns_ref, viscosity)
            Δt = min(Δt, tstop - t)
            right_hand_side!(cache_dns.du, u.dns_ref; viscosity)
            fe_step!(cache_dns.du, u.dns_ref, cache_dns, Δt, poisson_dns)

            # No-model
            right_hand_side!(cache_les.du, u.nomodel; viscosity)
            fe_step!(cache_les.du, u.nomodel, cache_les, Δt, poisson_les)

            # Classic
            right_hand_side!(cache_les.du, u.classic; viscosity)
            σ_classic = LazyTensorField(
                g_les,
                (fru, rfu, i, j, I) -> fru[i, j][I] - rfu[i, j][I],
                fru,
                rfu,
            )
            apply!(tensordivergence!, g_les, cache_les.du, σ_classic)
            fe_step!(cache_les.du, u.classic, cache_les, Δt, poisson_les)

            # Swap-filter
            right_hand_side!(cache_les.du, u.swapfil; viscosity)
            σ_swapfil = LazyTensorField(
                g_les,
                (firu, rfu, i, j, I) -> firu[i, j][I] - rfu[i, j][I],
                firu,
                rfu,
            )
            apply!(tensordivergence!, g_les, cache_les.du, σ_swapfil)
            fe_step!(cache_les.du, u.swapfil, cache_les, Δt, poisson_les)

            # Swap-filter-symmetrized
            right_hand_side!(cache_les.du, u.swapfil_symm; viscosity)
            σ_swapfil_symm = LazyTensorField(
                g_les,
                (σ, i, j, I) -> (σ[i, j][I] + σ[j, i][I]) / 2,
                σ_swapfil,
            )
            apply!(tensordivergence!, g_les, cache_les.du, σ_swapfil_symm)
            fe_step!(cache_les.du, u.swapfil_symm, cache_les, Δt, poisson_les)

            # Filtered DNS. Must be last! Otherwise it will modify the lazy
            # tensors that depend on u.dns_fil
            Turbulox.volumefilter!(u.dns_fil, u.dns_ref, compression)

            # Update time
            t += Δt
        end
        push!(relerr.time, t)
        push!(relerr.nomodel, norm(u.nomodel.data - u.dns_fil.data) / norm(u.dns_fil.data))
        push!(relerr.classic, norm(u.classic.data - u.dns_fil.data) / norm(u.dns_fil.data))
        push!(relerr.swapfil, norm(u.swapfil.data - u.dns_fil.data) / norm(u.dns_fil.data))
        push!(
            relerr.swapfil_symm,
            norm(u.swapfil_symm.data - u.dns_fil.data) / norm(u.dns_fil.data),
        )
        i += 1
        @show t
    end
    Turbulox.volumefilter!(u.dns_fil, u.dns_ref, compression)
    u, relerr
end

function dns_aid_volavg(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject,
)
    p_dns = ScalarField(g_dns)
    p_les = ScalarField(g_les)
    du_dns = VectorField(g_dns)
    du_les = VectorField(g_les)
    dσ_classic = VectorField(g_les)
    dσ_swapfil = VectorField(g_les)
    dσ_swapfil_symm = VectorField(g_les)
    x, y, z = X(), Y(), Z()
    T = typeof(g_dns.L)
    udns = VectorField(g_dns, copy(ustart.data))
    t = 0 |> T
    cfl = 0.15 |> T
    tstop = 0.1 |> T
    fru = TensorField(g_les)
    firu = TensorField(g_les)
    fu = VectorField(g_les)
    Turbulox.volumefilter!(fu, udns, compression)
    doproject && project!(fu, p_les, poisson_les)
    u = (;
        dns_ref = udns,
        dns_fil = fu,
        nomodel = VectorField(g_les, copy(fu.data)),
        classic = VectorField(g_les, copy(fu.data)),
        swapfil = VectorField(g_les, copy(fu.data)),
        swapfil_symm = VectorField(g_les, copy(fu.data)),
    )
    relerr = (;
        time = zeros(0),
        nomodel = zeros(0),
        classic = zeros(0),
        swapfil = zeros(0),
        swapfil_symm = zeros(0),
    )
    i = 0
    while t < tstop
        # Skip first iter to get initial errors
        if i > 0

            # Time step from current DNS
            Δt = cfl * propose_timestep(u.dns_ref, viscosity)
            Δt = min(Δt, tstop - t)

            # Sub-filter stress components
            su = stresstensor(u.dns_ref, viscosity)
            fill!(du_dns.data, 0)
            apply!(tensordivergence!, g_dns, du_dns, su)
            apply!(divergence!, g_dns, p_dns, du_dns)
            poissonsolve!(p_dns, poisson_dns)
            apply!(pressuregradient!, g_dns, du_dns, p_dns)
            @inline rukernel(u, p, viscosity, i, j, I) =
                stress(u, viscosity, i, j, I) + (i == j) * p[I] # Inline this
            ru = LazyTensorField(g_dns, rukernel, u.dns_ref, p_dns, viscosity)
            Turbulox.volumefilter!(fru, ru, compression)
            Turbulox.surfacefilter!(firu, ru, compression, false)
            Turbulox.volumefilter!(fu, u.dns_ref, compression)
            doproject && project!(fu, p_les, poisson_les)
            sfu = stresstensor(fu, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, sfu)
            apply!(divergence!, g_les, p_les, du_les)
            poissonsolve!(p_les, poisson_les)
            rfu = LazyTensorField(
                g_les,
                (u, p, viscosity, i, j, I) ->
                    stress(u, viscosity, i, j, I) + (i == j) * p[I],
                fu,
                p_les,
                viscosity,
            )
            σ_classic = LazyTensorField(
                g_les,
                (fru, rfu, i, j, I) -> fru[i, j][I] - rfu[i, j][I],
                fru,
                rfu,
            )
            σ_swapfil = LazyTensorField(
                g_les,
                (firu, rfu, i, j, I) -> firu[i, j][I] - rfu[i, j][I],
                firu,
                rfu,
            )
            σ_swapfil_symm = LazyTensorField(
                g_les,
                (σ, i, j, I) -> (σ[i, j][I] + σ[j, i][I]) / 2,
                σ_swapfil,
            )
            fill!(dσ_classic.data, 0)
            fill!(dσ_swapfil.data, 0)
            fill!(dσ_swapfil_symm.data, 0)
            apply!(tensordivergence!, g_les, dσ_classic, σ_classic)
            apply!(tensordivergence!, g_les, dσ_swapfil, σ_swapfil)
            apply!(tensordivergence!, g_les, dσ_swapfil_symm, σ_swapfil_symm)

            # DNS step
            axpy!(Δt, du_dns.data, u.dns_ref.data)

            # No-model
            σ_nomodel = LazyTensorField(
                g_les,
                (ubar, i, j, I) -> stress(ubar, viscosity, i, j, I),
                u.nomodel,
            )
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, σ_nomodel)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.nomodel.data)
            # This thing is already projected

            # Classic
            s = stresstensor(u.classic, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, s)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.classic.data)
            axpy!(Δt, dσ_classic.data, u.classic.data) # closure
            doproject && project!(u.classic, p_les, poisson_les)

            # Swap-filter
            s = stresstensor(u.swapfil, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, s)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.swapfil.data)
            axpy!(Δt, dσ_swapfil.data, u.swapfil.data) # closure
            doproject && project!(u.swapfil, p_les, poisson_les)

            # Swap-filter-symmetrized
            s = stresstensor(u.swapfil_symm, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, s)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.swapfil_symm.data)
            axpy!(Δt, dσ_swapfil_symm.data, u.swapfil_symm.data) # closure
            doproject && project!(u.swapfil_symm, p_les, poisson_les)

            # Time
            t += Δt
        end
        Turbulox.volumefilter!(fu, u.dns_ref, compression)
        doproject && project!(fu, p_les, poisson_les)
        push!(relerr.time, t)
        push!(relerr.nomodel, norm(u.nomodel.data - fu.data) / norm(fu.data))
        push!(relerr.classic, norm(u.classic.data - fu.data) / norm(fu.data))
        push!(relerr.swapfil, norm(u.swapfil.data - fu.data) / norm(fu.data))
        push!(relerr.swapfil_symm, norm(u.swapfil_symm.data - fu.data) / norm(fu.data))
        i += 1
        @show t
    end
    Turbulox.volumefilter!(u.dns_fil, u.dns_ref, compression)
    doproject && project!(u.dns_fil, p_les, poisson_les)
    u, relerr
end

function dns_aid_surface(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject,
)
    p_dns = ScalarField(g_dns)
    p_les = ScalarField(g_les)
    du_dns = VectorField(g_dns)
    du_les = VectorField(g_les)
    dσ_classic = VectorField(g_les)
    dσ_swapfil = VectorField(g_les)
    dσ_swapfil_symm = VectorField(g_les)
    x, y, z = X(), Y(), Z()
    T = typeof(g_dns.L)
    udns = VectorField(g_dns, copy(ustart.data))
    t = 0 |> T
    cfl = 0.15 |> T
    tstop = 0.1 |> T
    firu = TensorField(g_les)
    fijru = TensorField(g_les)
    fiu = VectorField(g_les)
    Turbulox.surfacefilter!(fiu, udns, compression)
    u = (;
        dns_ref = udns,
        dns_fil = fiu,
        nomodel = VectorField(g_les, copy(fiu.data)),
        classic = VectorField(g_les, copy(fiu.data)),
        swapfil = VectorField(g_les, copy(fiu.data)),
        swapfil_symm = VectorField(g_les, copy(fiu.data)),
    )
    relerr = (;
        time = zeros(0),
        nomodel = zeros(0),
        classic = zeros(0),
        swapfil = zeros(0),
        swapfil_symm = zeros(0),
    )
    i = 0
    while t < tstop
        # Skip first iter to get initial errors
        if i > 0

            # Time step from current DNS
            Δt = cfl * propose_timestep(u.dns_ref, viscosity)
            Δt = min(Δt, tstop - t)

            # Sub-filter stress components
            su = stresstensor(u.dns_ref, viscosity)
            fill!(du_dns.data, 0)
            apply!(tensordivergence!, g_dns, du_dns, su)
            apply!(divergence!, g_dns, p_dns, du_dns)
            poissonsolve!(p_dns, poisson_dns)
            apply!(pressuregradient!, g_dns, du_dns, p_dns)
            @inline rukernel(u, p, viscosity, i, j, I) =
                stress(u, viscosity, i, j, I) + (i == j) * p[I] # Inline this
            ru = LazyTensorField(g_dns, rukernel, u.dns_ref, p_dns, viscosity)
            Turbulox.surfacefilter!(firu, ru, compression, true)
            Turbulox.linefilter!(fijru, ru, compression)
            # error()
            Turbulox.surfacefilter!(fiu, u.dns_ref, compression)
            sfiu = stresstensor(fiu, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, sfiu)
            apply!(divergence!, g_les, p_les, du_les)
            poissonsolve!(p_les, poisson_les)
            rfiu = LazyTensorField(
                g_les,
                (u, p, viscosity, i, j, I) ->
                    stress(u, viscosity, i, j, I) + (i == j) * p[I],
                fiu,
                p_les,
                viscosity,
            )
            σ_classic = LazyTensorField(
                g_les,
                (firu, rfiu, i, j, I) -> firu[i, j][I] - rfiu[i, j][I],
                firu,
                rfiu,
            )
            σ_swapfil = LazyTensorField(
                g_les,
                (firu, fijru, rfiu, i, j, I) ->
                    (i == j) * firu[i, j][I] + (i != j) * fijru[i, j][I] - rfiu[i, j][I],
                firu,
                fijru,
                rfiu,
            )
            σ_swapfil_symm = LazyTensorField(
                g_les,
                (σ, i, j, I) -> (σ[i, j][I] + σ[j, i][I]) / 2,
                σ_swapfil,
            )
            fill!(dσ_classic.data, 0)
            fill!(dσ_swapfil.data, 0)
            fill!(dσ_swapfil_symm.data, 0)
            apply!(tensordivergence!, g_les, dσ_classic, σ_classic)
            apply!(tensordivergence!, g_les, dσ_swapfil, σ_swapfil)
            apply!(tensordivergence!, g_les, dσ_swapfil_symm, σ_swapfil_symm)

            # DNS step
            axpy!(Δt, du_dns.data, u.dns_ref.data)

            # No-model
            σ_nomodel = LazyTensorField(
                g_les,
                (ubar, i, j, I) -> stress(ubar, viscosity, i, j, I),
                u.nomodel,
            )
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, σ_nomodel)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.nomodel.data)
            # This thing is already projected

            # Classic
            s = stresstensor(u.classic, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, s)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.classic.data)
            axpy!(Δt, dσ_classic.data, u.classic.data) # closure
            doproject && project!(u.classic, p_les, poisson_les)

            # Swap-filter
            s = stresstensor(u.swapfil, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, s)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.swapfil.data)
            axpy!(Δt, dσ_swapfil.data, u.swapfil.data) # closure
            doproject && project!(u.swapfil, p_les, poisson_les)

            # Swap-filter-symmetrized
            s = stresstensor(u.swapfil_symm, viscosity)
            fill!(du_les.data, 0)
            apply!(tensordivergence!, g_les, du_les, s)
            project!(du_les, p_les, poisson_les)
            axpy!(Δt, du_les.data, u.swapfil_symm.data)
            axpy!(Δt, dσ_swapfil_symm.data, u.swapfil_symm.data) # closure
            doproject && project!(u.swapfil_symm, p_les, poisson_les)

            # Time
            t += Δt
        end
        Turbulox.surfacefilter!(fiu, u.dns_ref, compression)
        push!(relerr.time, t)
        push!(relerr.nomodel, norm(u.nomodel.data - fiu.data) / norm(fiu.data))
        push!(relerr.classic, norm(u.classic.data - fiu.data) / norm(fiu.data))
        push!(relerr.swapfil, norm(u.swapfil.data - fiu.data) / norm(fiu.data))
        push!(relerr.swapfil_symm, norm(u.swapfil_symm.data - fiu.data) / norm(fiu.data))
        i += 1
        @show t
    end
    Turbulox.surfacefilter!(u.dns_fil, u.dns_ref, compression)
    u, relerr
end

function sfs_tensors_volume(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject,
)
    p_dns = ScalarField(g_dns)
    p_les = ScalarField(g_les)
    du_dns = VectorField(g_dns)
    du_les = VectorField(g_les)
    T = typeof(g_dns.L)
    u = ustart
    fru = TensorField(g_les)
    firu = TensorField(g_les)
    fu = VectorField(g_les)
    Turbulox.volumefilter!(fu, u, compression)
    doproject && project!(fu, p_les, poisson_les)

    su = stresstensor(u, viscosity)
    fill!(du_dns.data, 0)
    apply!(tensordivergence!, g_dns, du_dns, su)
    apply!(divergence!, g_dns, p_dns, du_dns)
    poissonsolve!(p_dns, poisson_dns)
    @inline rukernel(u, p, viscosity, i, j, I) =
        stress(u, viscosity, i, j, I) + (i == j) * p[I] # Inline this
    ru = LazyTensorField(g_dns, rukernel, u, p_dns, viscosity)
    Turbulox.volumefilter!(fru, ru, compression)
    Turbulox.surfacefilter!(firu, ru, compression, false)
    Turbulox.volumefilter!(fu, u, compression)
    doproject && project!(fu, p_les, poisson_les)
    sfu = stresstensor(fu, viscosity)
    fill!(du_les.data, 0)
    apply!(tensordivergence!, g_les, du_les, sfu)
    apply!(divergence!, g_les, p_les, du_les)
    poissonsolve!(p_les, poisson_les)
    rfu = LazyTensorField(
        g_les,
        (u, p, viscosity, i, j, I) -> stress(u, viscosity, i, j, I) + (i == j) * p[I],
        fu,
        p_les,
        viscosity,
    )
    σ_classic =
        LazyTensorField(g_les, (fru, rfu, i, j, I) -> fru[i, j][I] - rfu[i, j][I], fru, rfu)
    σ_swapfil = LazyTensorField(
        g_les,
        (firu, rfu, i, j, I) -> firu[i, j][I] - rfu[i, j][I],
        firu,
        rfu,
    )
    σ_swapfil_symm =
        LazyTensorField(g_les, (σ, i, j, I) -> (σ[i, j][I] + σ[j, i][I]) / 2, σ_swapfil)

    (; σ_classic, σ_swapfil, σ_swapfil_symm)
end

function sfs_tensors_surface(;
    ustart,
    g_dns,
    g_les,
    poisson_dns,
    poisson_les,
    viscosity,
    compression,
    doproject,
)
    p_dns = ScalarField(g_dns)
    p_les = ScalarField(g_les)
    du_dns = VectorField(g_dns)
    du_les = VectorField(g_les)
    T = typeof(g_dns.L)
    u = ustart
    firu = TensorField(g_les)
    fijru = TensorField(g_les)
    fiu = VectorField(g_les)
    Turbulox.surfacefilter!(fiu, u, compression)

    # Sub-filter stress components
    su = stresstensor(u, viscosity)
    fill!(du_dns.data, 0)
    apply!(tensordivergence!, g_dns, du_dns, su)
    apply!(divergence!, g_dns, p_dns, du_dns)
    poissonsolve!(p_dns, poisson_dns)
    @inline rukernel(u, p, viscosity, i, j, I) =
        stress(u, viscosity, i, j, I) + (i == j) * p[I] # Inline this
    ru = LazyTensorField(g_dns, rukernel, u, p_dns, viscosity)
    surfacefilter!(firu, ru, compression, true)
    linefilter!(fijru, ru, compression)
    surfacefilter!(fiu, u, compression)
    sfiu = stresstensor(fiu, viscosity)
    fill!(du_les.data, 0)
    apply!(tensordivergence!, g_les, du_les, sfiu)
    apply!(divergence!, g_les, p_les, du_les)
    poissonsolve!(p_les, poisson_les)
    rfiu = LazyTensorField(
        g_les,
        (u, p, viscosity, i, j, I) -> stress(u, viscosity, i, j, I) + (i == j) * p[I],
        fiu,
        p_les,
        viscosity,
    )
    σ_classic = LazyTensorField(
        g_les,
        (firu, rfiu, i, j, I) -> firu[i, j][I] - rfiu[i, j][I],
        firu,
        rfiu,
    )
    σ_swapfil = LazyTensorField(
        g_les,
        (firu, fijru, rfiu, i, j, I) ->
            (i == j) * firu[i, j][I] + (i != j) * fijru[i, j][I] - rfiu[i, j][I],
        firu,
        fijru,
        rfiu,
    )
    σ_swapfil_symm =
        LazyTensorField(g_les, (σ, i, j, I) -> (σ[i, j][I] + σ[j, i][I]) / 2, σ_swapfil)

    (; σ_classic, σ_swapfil, σ_swapfil_symm)
end

end
