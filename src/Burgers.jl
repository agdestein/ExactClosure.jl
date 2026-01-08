"1D simulation."
module Burgers

using ..ExactClosure

using Adapt
using CairoMakie
using CUDA
using FFTW
using JET
using JLD2
using KernelAbstractions
using LinearAlgebra
using Makie
using Printf
using Random
import AcceleratedKernels as AK

defaultbackend() = CUDA.functional() ? CUDABackend() : KernelAbstractions.CPU()

myforeachindex(f::F, data) where {F} = AK.foreachindex(f, data)
# myforeachindex(f::F, data) where {F} = foreach(f, eachindex(data))

"Uniform grid of length `L` and `n` volumes."
struct Grid{T}
    L::T
    n::Int
end

"Grid spacing."
@inline spacing(g::Grid) = g.L / g.n

struct Field{T,A} <: AbstractArray{T, 1}
    grid::Grid{T}
    data::A
end

Field(g::Grid, b::Backend) =
    Field(g, KernelAbstractions.zeros(b, typeof(g.L), g.n))

struct LazyField{F, T, S} <: AbstractArray{T, 1}
    func::F
    grid::Grid{T}
    stuff::S
end

Adapt.adapt_structure(to, u::Field) = Field(u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::LazyField) =
    LazyField(adapt(to, u.func), u.grid, adapt(to, u.stuff))

@inline Base.size(u::Field) = (u.grid.n,)
@inline Base.size(u::LazyField) = (u.grid.n,)

@inline @inbounds Base.getindex(u::Field, i::Int) =
    getindex(u.data, mod1(i, u.grid.n))
@inline @inbounds Base.getindex(u::LazyField{F}, i::Int) where {F} =
    u.func(u.stuff..., i)

@inline @inbounds Base.setindex!(u::Field, val, i::Int) =
    setindex!(u.data, val, mod1(i, u.grid.n))

Base.show(io::IO, u::Field) = print(io, "Field($(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::LazyField) = print(
    io,
    "LazyField(",
    join((u.func, u.grid, map(s -> "::$(typeof(s))", u.stuff)), ", ")...,
    ")",
)
Base.show(io::IO, ::MIME"text/plain", u::Field) =
    print(io, join(map(string, size(u)), "×")..., " ", u)
Base.show(io::IO, ::MIME"text/plain", u::LazyField) =
    print(io, join(map(string, size(u)), "×")..., " ", u)

function Base.similar(u::Field, ::Type{T}, dims::Dims) where {T}
    @assert dims == size(u.data) "Scalar field must have same size as grid."
    Field(u.grid, similar(u.data, T))
end

Base.copyto!(v::Field, u::Field) = (copyto!(v.data, u.data); v)

# Grid points
points_stag(g) = range(0, g.L, g.n + 1)[2:end]
points_coll(g) = range(0, g.L, g.n + 1)[2:end] .- spacing(g) / 2

# Finite difference
@inline @inbounds δ_stag(u, i) = (u[i] - u[i-1]) / spacing(u.grid)
@inline @inbounds δ_coll(p, i) = (p[i+1] - p[i]) / spacing(p.grid)

stress(u, visc) = LazyField(u.grid, (u, visc)) do u, visc, i
    @inline
    @inbounds ua = u[i-1]
    @inbounds ub = u[i]
    -(ua + ub)^2 / 8 + visc * (ub - ua) / spacing(u.grid)
end

"CFL number."
cfl(u, visc) = let
    g = u.grid
    h = spacing(g)
    min(h / maximum(abs, u.data), h^2 / visc)
end

rhs!(du, u, visc) = let
    s = stress(u, visc)
    myforeachindex(u.data) do i
        @inline
        du[i] = δ_coll(s, i)
    end
end

"Perform one time step (inplace)."
timestep!(u, visc, dt) = let
    s = stress(u, visc)
    myforeachindex(u.data) do i
        @inline
        @inbounds u[i] += dt * δ_coll(s, i)
    end
end

function rk4_timestep!(u, uold, k1, k2, k3, k4, visc, dt)
    copyto!(uold, u)
    rhs!(k1, u, visc)
    myforeachindex(u) do i
        @inline
        @inbounds u[i] = uold[i] + dt / 2 * k1[i]
    end
    rhs!(k2, u, visc)
    myforeachindex(u) do i
        @inline
        @inbounds u[i] = uold[i] + dt / 2 * k2[i]
    end
    rhs!(k3, u, visc)
    myforeachindex(u) do i
        @inline
        @inbounds u[i] = uold[i] + dt * k3[i]
    end
    rhs!(k4, u, visc)
    myforeachindex(u) do i
        @inline
        @inbounds u[i] = uold[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    end
end

# Filters

function gaussian_weights(g, Δ; nσ = 3)
    σ2 = (Δ / spacing(g))^2 / 12
    R = round(Int, nσ * sqrt(σ2))
    if Δ < 1e-15
        w = [one(Δ)]
    else
        w = map(r -> exp(-r^2 / 2σ2), (-R):R)
        w ./= sum(w)
    end
    R, w
end

function tophat_weights(g, comp)
    R = div(comp, 2)
    w = fill(one(g.L) / comp, comp)
    R, w
end

lazyfilter(u, kernel) = LazyField(u.grid, (u, kernel)) do u, kernel, i
    @inline
    R, w = kernel
    s = zero(eltype(u))
    for r = (-R):R
        @inbounds s += w[r+R+1] * u[i+r]
    end
    s
end

lazycoarsegrain(gbar, u, stag) = LazyField(gbar, (u, stag)) do u, stag, i
    @inline
    g = u.grid
    comp = div(g.n, gbar.n)
    a = div(comp, 2)
    @inbounds u[i * comp - !(stag) * a]
end

materialize!(u::Field, lu::LazyField) = let
    g = u.grid
    lg = lu.grid
    @assert g == lg
    myforeachindex(u.data) do i
        @inline
        @inbounds u.data[i] = lu[i]
    end
end

bump_profile(rng, k, kpeak) = (k / kpeak)^4 * exp(-2 * (k / kpeak)^2 + 2π * im * rand(rng))
linear_profile(rng, k, kpeak) = k > 0 ? k^-2.0 * exp(2π * im * rand(rng)) : 0.0 + 0.0im

function randomfield(rng, g, kpeak, totalenergy, backend)
    k = 0:div(g.n, 2)
    c = map(k -> bump_profile(rng, k, kpeak), k)
    a = irfft(c * g.n, g.n)
    etot = sum(abs2, a) / 2 * spacing(g)
    u = sqrt(totalenergy / etot) * a
    Field(g, adapt(backend, u))
end

dissipation(g, u, s) =
    map(1:g.n) do i
        δ_stag(g, u, i) * s[i]
        # u[i] * δ_coll(g, s, i)
    end

function dissipation!(g, d, u, s)
    myforeachindex(d) do i
        @inline
        @inbounds d[i] = s[i] * δ_stag(g, u, i)
    end
end

"Combine two filter kernels into a double-filter kernel."
function build_doublekernel(F, G)
    I, f = F
    J, g = G
    K = I + J
    w_double = map((-K):K) do k
        sum((-J):J) do j
            i = k - j
            if abs(i) ≤ I
                f[I+1+i] * g[J+1+j]
            else
                zero(eltype(f))
            end
        end
    end
    K, w_double
end

getsetup() = let
    L = 2π
    nh = 100 * 3^3 * 5
    nH = 100 .* 3 .^ (1:3)
    # Δ_ratios = [0, 1, 2]
    Δ_ratios = [2]
    visc = 5e-4
    kpeak = 10
    initialenergy = 2.0
    tstop = 0.2
    nsample = 100O
    lesfiltertype = :gaussian
    backend = CUDA.functional() ? CUDABackend() : KernelAbstractions.CPU()
    outdir = joinpath(@__DIR__, "..", "output", "Burgers") |> mkpath
    plotdir = "$outdir/figures" |> mkpath
    (;
        L,
        nh,
        nH,
        Δ_ratios,
        visc,
        kpeak,
        initialenergy,
        tstop,
        nsample,
        lesfiltertype,
        backend,
        outdir,
        plotdir,
    )
end

run_dns_aided_les(setup) =
    for (nH, Δ_ratio) in Iterators.product(setup.nH, setup.Δ_ratios)
        (; L, nh, visc, kpeak, initialenergy, tstop, nsample, backend, outdir) = setup
        cpu = KernelAbstractions.CPU()
        gh = Grid(L, nh)
        gH = Grid(L, nH)
        Δ = spacing(gH) * Δ_ratio
        fields = (;
            dns_ref = (; g = gh, u = zeros(nh, nsample) |> adapt(backend), label = "DNS"),
            dns_ref_fil = (;
                g = gh,
                u = zeros(nh, nsample) |> adapt(backend),
                label = "Filtered DNS",
            ),
            dns_fil = (;
                g = gH,
                u = zeros(nH, nsample) |> adapt(backend),
                label = "Filtered DNS",
            ),
            nomodel = (;
                g = gH,
                u = zeros(nH, nsample) |> adapt(backend),
                label = "No model",
            ),
            class_m = (;
                g = gH,
                u = zeros(nH, nsample) |> adapt(backend),
                label = "Classic",
            ),
            class_p = (;
                g = gH,
                u = zeros(nH, nsample) |> adapt(backend),
                label = "Classic+Flux",
            ),
            swapfil = (;
                g = gH,
                u = zeros(nH, nsample) |> adapt(backend),
                label = "Classic+Flux+Div (ours)",
            ),
        )
        for i = 1:nsample
            @info "Δ/h = $Δ_ratio, N = $nH, sample $i of $nsample"
            rng = Xoshiro(i)
            ustart = randomfield(rng, gh, kpeak, initialenergy, backend)
            @time sols = dns_aided_les(
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
                copyto!(view(fields[key].u, :, i), sols[key].data)
            end
        end
        result = (; Δ_ratio, nH, fields = adapt(cpu, fields))
        file = "$outdir/burgers_Δ_ratio=$(Δ_ratio)_hH=$(nH).jld2"
        save_object(file, result)
    end

load_dns_aided_les(setup) =
    map(Iterators.product(setup.nH, setup.Δ_ratios)) do (nH, Δ_ratio)
        (; outdir) = setup
        file = "$outdir/burgers_Δ_ratio=$(Δ_ratio)_hH=$(nH).jld2"
        @info "Loading $file"
        load_object(file)
    end

stepfromstress!(u, du, s, dt) = let
    myforeachindex(du.data) do i
        @inline
        @inbounds du[i] = δ_coll(s, i)
    end
    myforeachindex(u.data) do i
        @inline
        @inbounds u[i] += dt * du[i]
    end
end

# Compare closure formulations
function dns_aided_les(ustart, gh, gH, visc; Δ, tstop, cfl_factor, lesfiltertype)
    backend = AK.get_backend(ustart.data)
    uh = ustart

    # Filter kernels
    fvmkernel = tophat_weights(gH, div(gh.n, gH.n))
    leskernel = if lesfiltertype == :tophat
        lescomp = round(Int, Δ / spacing(gh))
        R = div(lescomp, 2)
        lescomp = 2 * R + 1 # Ensure odd
        tophat_weights(gH, lescomp)
    elseif lesfiltertype == :gaussian
        gaussian_weights(gh, Δ)
    end
    doublekernel = build_doublekernel(fvmkernel, leskernel)

    # Put on device
    fvmkernel = fvmkernel |> adapt(backend)
    leskernel = leskernel |> adapt(backend)
    doublekernel = doublekernel |> adapt(backend)

    # Initial double-filtered state
    uH = Field(gH, backend)
    materialize!(uH, lazycoarsegrain(gH, lazyfilter(uh, doublekernel), true))

    # Allocate arrays
    u = (;
        dns_ref     = copy(uh),
        dns_ref_fil = copy(uh),
        dns_fil     = copy(uH),
        nomodel     = copy(uH),
        class_m     = copy(uH), # bar(r(u)) - r(bar(u))
        class_p     = copy(uH), # bar(r(u)) - rh(bar(u))
        swapfil     = copy(uH),
    )

    du = Field(gh, backend)
    duH = Field(gH, backend)
    vsu = Field(gH, backend)
    fsu = Field(gH, backend)
    vu = Field(gH, backend)
    sVU = Field(gH, backend)

    # Time stepping
    t = 0.0
    while t < tstop
        # Get time step
        dt = cfl_factor * cfl(u.dns_ref, visc)
        dt = min(dt, tstop - t) # Don't overstep

        # DNS stress
        su = stress(u.dns_ref, visc)

        # Filtered stresses
        # fsu = lazycoarsegrain(gH, lazyfilter(su, leskernel), false)
        # vsu = lazycoarsegrain(gH, lazyfilter(su, doublekernel), false)
        materialize!(fsu , lazycoarsegrain(gH, lazyfilter(su, leskernel), false))
        materialize!(vsu , lazycoarsegrain(gH, lazyfilter(su, doublekernel), false))

        # Filtered velocity
        VU = lazyfilter(u.dns_ref, doublekernel)
        svu = stress(vu, visc)
        # vu = lazycoarsegrain(gH, VU, true)
        # sVU = lazycoarsegrain(gH, stress(VU, visc), false)
        materialize!(vu , lazycoarsegrain(gH, VU, true))
        materialize!(sVU, lazycoarsegrain(gH, stress(VU, visc), false))

        # LES stresses
        s_nomodel = stress(u.nomodel, visc)
        s_class_m = stress(u.class_m, visc)
        s_class_p = stress(u.class_p, visc)
        s_swapfil = stress(u.swapfil, visc)

        σ_nomodel = s_nomodel
        σ_class_m = LazyField((s, vsu, sVU, i) -> (@inline @inbounds s[i] + vsu[i] - sVU[i]), gH, (s_class_m, vsu, sVU))
        σ_class_p = LazyField((s, vsu, svu, i) -> (@inline @inbounds s[i] + vsu[i] - svu[i]), gH, (s_class_p, vsu, svu))
        σ_swapfil = LazyField((s, fsu, svu, i) -> (@inline @inbounds s[i] + fsu[i] - svu[i]), gH, (s_swapfil, fsu, svu))

        # LES time steps
        stepfromstress!(u.nomodel, duH, σ_nomodel, dt)
        stepfromstress!(u.class_m, duH, σ_class_m, dt)
        stepfromstress!(u.class_p, duH, σ_class_p, dt)
        stepfromstress!(u.swapfil, duH, σ_swapfil, dt)

        # DNS time step (after LES since lazy fields depend on current DNS state)
        stepfromstress!(u.dns_ref, du, su, dt)

        # Time step
        t += dt
    end

    materialize!(u.dns_ref_fil, lazyfilter(u.dns_ref, doublekernel))
    materialize!(u.dns_fil, lazycoarsegrain(gH, u.dns_ref_fil, true))

    u
end

function compute_dissipation(series, setup)
    (; visc, lesfiltertype) = setup
    map(series) do (; nH, Δ_ratio, fields)
        gh = Grid(setup.L, setup.nh)
        gH = Grid(setup.L, nH)
        Δ = spacing(gH) * Δ_ratio

        # Filter kernels
        fvmkernel = tophat_weights(gH, div(gh.n, gH.n))
        leskernel = if lesfiltertype == :tophat
            lescomp = round(Int, Δ / spacing(gh))
            R = div(lescomp, 2)
            lescomp = 2 * R + 1 # Ensure odd
            tophat_weights(gH, lescomp)
        elseif lesfiltertype == :gaussian
            gaussian_weights(gh, Δ)
        end

        # Build double-filter kernel
        doublekernel = build_doublekernel(fvmkernel, leskernel)

        # Allocate arrays
        su = zeros(gh.n)
        fsu = zeros(gH.n)
        vsu = zeros(gH.n)
        vu = zeros(gH.n)
        VU = zeros(gh.n) # Same as vu, but on DNS grid
        σ_class_m = zeros(gH.n)
        σ_class_p = zeros(gH.n)
        σ_swapfil = zeros(gH.n)

        d = stack(eachcol(fields.dns_ref.u)) do uh
            # DNS stress
            myforeachindex(su) do i
                @inline
                @inbounds su[i] = stress(gh, uh, visc, i)
            end

            # Filtered stresses
            coarsegrain_convolve_coll!(gH, gh, fsu, su, leskernel)
            coarsegrain_convolve_coll!(gH, gh, vsu, su, doublekernel)

            # Filtered velocity
            coarsegrain_convolve_stag!(gH, gh, vu, uh, doublekernel)
            convolution!(gh, doublekernel, VU, uh)

            # LES stresses
            comp = div(gh.n, gH.n)
            a = div(comp, 2)
            myforeachindex(vu) do i
                @inline
                @inbounds svu = stress(gH, vu, visc, i)
                @inbounds sVU = stress(gh, VU, visc, i * comp - a)
                @inbounds σ_class_m[i] = vsu[i] - sVU
                @inbounds σ_class_p[i] = vsu[i] - svu
                @inbounds σ_swapfil[i] = fsu[i] - svu
            end

            # Dissipation coefficients
            d_class_m = dissipation(gH, vu, σ_class_m)
            d_class_p = dissipation(gH, vu, σ_class_p)
            d_swapfil = dissipation(gH, vu, σ_swapfil)

            hcat(d_class_m, d_class_p, d_swapfil)
        end

        (;
            nomodel = zeros(1),
            class_m = d[:, 1, :] |> vec,
            class_p = d[:, 2, :] |> vec,
            swapfil = d[:, 3, :] |> vec,
        )
    end
end

function create_dns(setup; cfl_factor)
    (; L, nh, kpeak, initialenergy, visc, tstop, nsample) = setup
    g = Grid(L, nh)
    Ustart = zeros(nh, nsample)
    U = zeros(nh, nsample)
    for i = 1:nsample
        @info "DNS sample $i of $nsample"
        ustart = randomfield(Xoshiro(i), g, kpeak, initialenergy)
        u = copy(ustart)
        s = zero(u)
        # uold, k1, k2, k3, k4 = zero(u), zero(u), zero(u), zero(u), zero(u)
        t = 0.0
        while t < tstop
            dt = cfl_factor * cfl(g, u, visc) # Propose timestep
            dt = min(dt, tstop - t) # Don't overstep
            timestep!(g, u, s, visc, dt) # Perform timestep
            # rk4_timestep!(g, u, uold, k1, k2, k3, k4, visc, dt)
            t += dt
        end
        Ustart[:, i] = ustart
        U[:, i] = u
    end
    Ustart, U
end

function smagorinsky_fields(setup, dnsdata; lesfiltertype)
    (; L, nh, nH, visc, Δ_ratio) = setup
    map(nH) do nH
        gh = Grid(L, nh)
        gH = Grid(L, nH)
        _, U = dnsdata
        # U, _ = dnsdata

        H = spacing(gH)
        Δ = H * Δ_ratio
        comp = div(gh.n, gH.n)
        a = div(comp, 2) # We have comp = 2a + 1

        # Filter kernels
        fvmkernel = tophat_weights(gH, comp)
        leskernel = if lesfiltertype == :tophat
            lescomp = round(Int, Δ / spacing(gh))
            R = div(lescomp, 2)
            lescomp = 2 * R + 1 # Ensure odd
            tophat_weights(gH, lescomp)
        elseif lesfiltertype == :gaussian
            gaussian_weights(gh, Δ)
        end
        doublekernel = build_doublekernel(fvmkernel, leskernel)

        # Allocate arrays
        σh = zeros(gh.n)
        σh_double1 = zeros(gh.n)
        σh_double2 = zeros(gh.n)
        uh_double = zeros(gh.n)

        sh = zeros(gh.n)
        th = zeros(gh.n)
        D_sh = zeros(gh.n)
        D_th = zeros(gh.n)

        sH = zeros(gh.n)
        tH = zeros(gh.n)
        D_sH = zeros(gh.n)
        D_tH = zeros(gh.n)

        Sh = zero(U)
        Th = zero(U)
        D_Sh = zero(U)
        D_Th = zero(U)

        SH = zero(U)
        TH = zero(U)
        D_SH = zero(U)
        D_TH = zero(U)

        Uh_double = zero(U)

        # Compute classical sub-filter stress (on DNS grid)
        foreach(axes(U, 2)) do j
            uh = view(U, :, j)

            # Compute sub-filter stress
            myforeachindex(uh) do i
                @inline
                σh[i] = stress(gh, uh, visc, i)
            end
            convolution!(gh, doublekernel, σh_double1, σh)
            convolution!(gh, doublekernel, uh_double, uh)
            myforeachindex(uh) do i
                @inline
                σh_double2[i] = stress(gh, uh_double, visc, i)
                th[i] = σh_double1[i] - σh_double2[i]
            end

            # Compute Smagorinsky stress
            myforeachindex(uh_double) do i
                @inline
                δu = δ_stag(gh, uh_double, i)
                sh[i] = -(Δ^2 + H^2) * abs(δu) * δu
            end

            # Compute dissipation coefficients
            myforeachindex(uh_double) do i
                @inline
                δu = δ_stag(gh, uh_double, i)
                D_sh[i] = sh[i] * δu # Smag
                D_th[i] = th[i] * δu # True
            end

            Sh[:, j] = sh
            Th[:, j] = th
            D_Sh[:, j] = D_sh
            D_Th[:, j] = D_th
            Uh_double[:, j] = uh_double
        end

        # Compute discretization-consistent sub-filter stress (on DNS grid)
        foreach(axes(U, 2)) do j
            uh = view(U, :, j)

            # Compute sub-filter stress
            myforeachindex(uh) do i
                @inline
                σh[i] = stress(gh, uh, visc, i)
            end
            convolution!(gh, leskernel, σh_double1, σh)
            convolution!(gh, doublekernel, uh_double, uh)
            myforeachindex(uh) do i
                @inline
                # Coarse-grid stress (on the fine grid)
                uleft = uh_double[i-a-1|>gh]
                uright = uh_double[i+a|>gh]
                σh_double2[i] =
                    -(uleft + uright)^2 / 8 + visc * (uright - uleft) / spacing(gH)
                tH[i] = σh_double1[i] - σh_double2[i]
            end

            # Compute Smagorinsky stress
            myforeachindex(uh_double) do i
                @inline
                # Coarse-grid derivative (on fine grid)
                uleft = uh_double[i-a-1|>gh]
                uright = uh_double[i+a|>gh]
                δu = (uright - uleft) / spacing(gH)
                sH[i] = -(Δ^2 + H^2) * abs(δu) * δu
            end

            # Compute dissipation coefficients
            myforeachindex(uh_double) do i
                @inline
                uleft = uh_double[i-a-1|>gh]
                uright = uh_double[i+a|>gh]
                δu = (uright - uleft) / spacing(gH)
                D_sH[i] = sH[i] * δu # Smag
                D_tH[i] = tH[i] * δu # True
            end

            SH[:, j] = sH
            TH[:, j] = tH
            D_SH[:, j] = D_sH
            D_TH[:, j] = D_tH
        end

        (; Sh, Th, D_Sh, D_Th, SH, TH, D_SH, D_TH, Ubar = Uh_double)
    end
end

function solve_smagorinsky(setup, dnsdata, smagcoeffs; lesfiltertype)
    (; L, nh, nH, visc, tstop, Δ_ratio) = setup
    map(enumerate(nH)) do (igrid, nH)
        gh = Grid(L, nh)
        gH = Grid(L, nH)
        H = spacing(gH)
        Δ = H * Δ_ratio
        θ = smagcoeffs[igrid]
        C = θ^2 * (Δ^2 + H^2)

        ustart, _ = dnsdata

        # Filter kernels
        fvmkernel = tophat_weights(gH, div(gh.n, gH.n))
        leskernel = if lesfiltertype == :tophat
            lescomp = round(Int, Δ / spacing(gh))
            R = div(lescomp, 2)
            lescomp = 2 * R + 1 # Ensure odd
            tophat_weights(gH, lescomp)
        elseif lesfiltertype == :gaussian
            gaussian_weights(gh, Δ)
        end
        doublekernel = build_doublekernel(fvmkernel, leskernel)

        ubar = zeros(gH.n)
        Ubar = zeros(gH.n, size(ustart, 2))

        ubar0 = zero(ubar)
        k1 = zero(ubar)
        k2 = zero(ubar)
        k3 = zero(ubar)
        k4 = zero(ubar)

        s = zeros(gH.n)

        for j in axes(ustart, 2)
            coarsegrain_convolve_stag!(gH, gh, ubar, ustart[:, j], doublekernel)

            @info "nH = $nH, sample $j of $(size(ustart, 2))"

            cfl_factor = 0.3
            t = 0.0
            while t < tstop
                dt = cfl_factor * cfl(gH, ubar, visc)
                dt = min(dt, tstop - t) # Don't overstep
                t += dt

                # # Forward Euler step
                # smagorinsky_rhs!(k1, u, s, gH, visc, C)
                # myforeachindex(ubar) do i
                #     @inline
                #     ubar[i] += dt * k1[i]
                # end

                # RK4 step
                copyto!(ubar0, ubar)
                smagorinsky_rhs!(k1, ubar, s, gH, visc, C)
                myforeachindex(ubar) do i
                    @inline
                    ubar[i] = ubar0[i] + dt / 2 * k1[i]
                end
                smagorinsky_rhs!(k2, ubar, s, gH, visc, C)
                myforeachindex(ubar) do i
                    @inline
                    ubar[i] = ubar0[i] + dt / 2 * k2[i]
                end
                smagorinsky_rhs!(k3, ubar, s, gH, visc, C)
                myforeachindex(ubar) do i
                    @inline
                    ubar[i] = ubar0[i] + dt * k3[i]
                end
                smagorinsky_rhs!(k4, ubar, s, gH, visc, C)
                myforeachindex(ubar) do i
                    @inline
                    ubar[i] = ubar0[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                end
            end
            Ubar[:, j] = ubar
        end

        Ubar
    end
end

function smagorinsky_rhs!(du, u, s, g, visc, C)
    myforeachindex(u) do i
        @inline
        δu = δ_stag(g, u, i)
        smag = -C * abs(δu) * δu
        s[i] = stress(g, u, visc, i) - smag
    end
    myforeachindex(u) do i
        @inline
        du[i] = δ_coll(g, s, i)
    end
end

# Compute spectra
compute_spectra(series, setup) =
    map(series) do (; Δ_ratio, nH, fields)
        @show Δ_ratio, nH
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
        (; nH, Δ_ratio, specs)
    end

# Plot spectra
plot_spectra(specseries, setup) =
    for (iΔ, Δ_ratio) in enumerate(setup.Δ_ratios)
        (; plotdir) = setup
        fig = Figure(; size = (400, 800))
        f = fig[1, 1] = GridLayout()
        ticks = 2 .^ (0:2:10)
        axes = map(specseries[:, iΔ] |> enumerate) do (i, specseries)
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
                logy = 3.0,
                relwidth = 0.45,
                relheight = 0.45,
            )
            styles = (;
                dns_ref = (; color = :black),
                dns_ref_fil = (; color = :black, linestyle = :dash),
                nomodel = (; color = Cycled(1)),
                class_m = (; color = Cycled(2)),
                class_p = (; color = Cycled(3)),
                swapfil = (; color = Cycled(4)),
            )
            for key in [:dns_ref, :dns_ref_fil, :nomodel, :class_m, :class_p, :swapfil]
                (; k, e, label) = specs[key]
                # At the end of the spectrum, there are too many points for plotting.
                # Choose a logarithmically equispaced subset of points instead
                npoint = 500
                ii = round.(Int, logrange(1, length(k), npoint)) |> unique
                lines!(ax, k[ii], e[ii]; label, styles[key]...)
                # lines!(ax_zoom, k[ii], e[ii]; styles[key]...)
                lines!(ax_zoom, k, e; styles[key]...)
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
            ylims!(ax, 1e-12, 1e-1)
            ax
        end
        Legend(
            fig[0, 1],
            axes[1];
            tellwidth = false,
            orientation = :horizontal,
            nbanks = 3,
            framevisible = false,
        )
        linkaxes!(axes...)
        rowgap!(fig.layout, 10)
        save(
            "$(plotdir)/burgers_spectrum_Delta_ratio=$Δ_ratio.pdf",
            fig;
            backend = CairoMakie,
        )
        display(fig)
    end

compute_errors(series) =
    map([:nomodel, :class_m, :class_p, :swapfil]) do key
        e = map(series) do (; nH, fields)
            (; u) = fields[key]
            norm(u - fields.dns_fil.u) / norm(fields.dns_fil.u)
        end
        key => (; e, series[1].fields[key].label)
    end |> NamedTuple

# Write errors to LaTeX table
function write_error_table(errseries, setup)
    (; outdir) = setup
    path = joinpath(outdir, "tables") |> mkpath
    file = joinpath(path, "burgers_error.tex")
    open(file, "w") do io
        tab = "    "
        c = join(fill("r", length(errseries) + 2), " ")
        println(io, "\\begin{tabular}{$c}")
        println(io, tab, "\\toprule")
        labels = join(map(f -> f.label, errseries), " & ")
        println(io, tab, "\$\\Delta / h\$ & \$N\$ & $labels \\\\")
        println(io, tab, "\\midrule")
        for j in eachindex(setup.Δ_ratios), i in eachindex(setup.nH)
            e = map(f -> round(f.e[i, j]; sigdigits = 3), errseries)
            println(
                io,
                tab,
                setup.Δ_ratios[j],
                " & ",
                setup.nH[i],
                " & ",
                join(e, " & "),
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

# Plot dissipation coefficient density
function plot_dissipation(diss, setup)
    (; plotdir) = setup
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

function compute_fractions(data, setup)
    (; L, nh, visc, lesfiltertype) = setup
    map(Iterators.product(setup.nH, setup.Δ_ratios)) do (nH, Δ_ratio)
        gh = Grid(L, nh)
        gH = Grid(L, nH)
        Δ = spacing(gH) * Δ_ratio

        # Filter kernels
        fvmkernel = tophat_weights(gH, div(gh.n, gH.n))
        leskernel = if lesfiltertype == :tophat
            lescomp = round(Int, Δ / spacing(gh))
            R = div(lescomp, 2)
            lescomp = 2 * R + 1 # Ensure odd
            tophat_weights(gH, lescomp)
        elseif lesfiltertype == :gaussian
            gaussian_weights(gh, Δ)
        end

        # Build double-filter kernel
        doublekernel = build_doublekernel(fvmkernel, leskernel)

        # Allocate arrays
        su = zeros(gh.n)
        fsu = zeros(gH.n)
        vsu = zeros(gH.n)
        vu = zeros(gH.n)
        VU = zeros(gh.n) # Same as vu, but on DNS grid
        σ_classic = zeros(gH.n)
        σ_flux = zeros(gH.n)
        σ_div = zeros(gH.n)

        f_classic = zeros(gH.n)
        f_flux = zeros(gH.n)
        f_div = zeros(gH.n)

        frac = stack(eachcol(data[2])) do uh
            # DNS stress
            myforeachindex(su) do i
                @inline
                @inbounds su[i] = stress(gh, uh, visc, i)
            end

            # Filtered stresses
            coarsegrain_convolve_coll!(gH, gh, fsu, su, leskernel)
            coarsegrain_convolve_coll!(gH, gh, vsu, su, doublekernel)

            # Filtered velocity
            coarsegrain_convolve_stag!(gH, gh, vu, uh, doublekernel)
            convolution!(gh, doublekernel, VU, uh)

            # LES stresses
            comp = div(gh.n, gH.n)
            a = div(comp, 2)
            myforeachindex(vu) do i
                @inline
                @inbounds svu = stress(gH, vu, visc, i)
                @inbounds sVU = stress(gh, VU, visc, i * comp - a)
                @inbounds σ_classic[i] = vsu[i] - sVU
                @inbounds σ_flux[i] = sVU - svu
                @inbounds σ_div[i] = fsu[i] - vsu[i]

                @inbounds σ_total =
                    abs(σ_classic[i]) + abs(σ_flux[i]) + abs(σ_div[i]) + eps()

                @inbounds f_classic[i] = abs(σ_classic[i]) / σ_total
                @inbounds f_flux[i] = abs(σ_flux[i]) / σ_total
                @inbounds f_div[i] = abs(σ_div[i]) / σ_total
            end

            # # Fractions
            # frac_classic = norm(σ_classic) / norm(σ_classic .+ σ_flux .+ σ_div)
            # frac_flux = norm(σ_flux) / norm(σ_classic .+ σ_flux .+ σ_div)
            # frac_div = norm(σ_div) / norm(σ_classic .+ σ_flux .+ σ_div)

            # Fractions
            frac_classic = sum(f_classic) / gH.n
            frac_flux = sum(f_flux) / gH.n
            frac_div = sum(f_div) / gH.n

            vcat(frac_classic, frac_flux, frac_div)
        end

        frac = sum(frac; dims = 2) / size(frac, 2)

        (; classic = frac[1], flux = frac[2], div = frac[3])
    end
end

function plot_fractions(fractions, setup)
    fig = Figure(; size = (400, 800))
    (; Δ_ratios, nH, plotdir) = setup
    colors = Makie.wong_colors()
    for (i, nH) in enumerate(nH)
        f = fractions[i, :]
        f_classic = getindex.(f, :classic)
        f_flux = getindex.(f, :flux)
        f_div = getindex.(f, :div)
        fs = hcat(f_classic, f_flux, f_div)[:]
        nratio = length(Δ_ratios)
        ratios = repeat(1:nratio, 1, 3)
        group = hcat(fill(1, nratio), fill(2, nratio), fill(3, nratio))
        ax = Axis(
            fig[i, 1];
            ylabel = "Fraction",
            title = "N = $nH",
            xlabel = "Δ / h",
            xticks = (1:nratio, map(string, Δ_ratios)),
            xticklabelsvisible = i == length(setup.nH),
            xlabelvisible = i == length(setup.nH),
        )

        bar_labels = map(fs) do f
            # f = round(f, digits = 2)
            s = @sprintf "%.2f" f
            f < 0.031 ? "" : s
        end

        barplot!(
            ax,
            ratios[:],
            fs;
            color = colors[group[:]],
            stack = group[:],
            bar_labels,
            label_position = :center,

            # flip_labels_at=(0.0, 0.0),
        )
    end
    labels = ["Classic", "Flux", "Div"]
    elements = map(i -> PolyElement(; polycolor = colors[i]), eachindex(labels))
    title = "Residual flux part"
    Legend(
        fig[0, :],
        elements,
        labels,
        title;
        tellwidth = false,
        orientation = :horizontal,
        nbanks = 1,
        framevisible = false,
    )
    file = "$(plotdir)/burgers_fractions.pdf"
    save(file, fig; backend = CairoMakie)
    fig
end

end
