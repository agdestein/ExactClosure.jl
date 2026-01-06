module NavierStokes

using AcceleratedKernels: AcceleratedKernels as AK
using Adapt
using CairoMakie
using CUDA
using FFTW
using KernelAbstractions
using LinearAlgebra
using Random

myforeachindex(f, data) = AK.foreachindex(f, data)

defaultbackend() = CUDA.functional() ? CUDABackend() : KernelAbstractions.CPU()

struct Grid{D}
    l::Float64
    n::Int
end

@inline Grid(::Val{D}, l, n) where {D} = Grid{D}(l, n)

@inline linear2cartesian(g, i) = CartesianIndices(ntuple(Returns(g.n), dim(g)))[i]
@inline dim(::Grid{D}) where {D} = D
@inline spacing(g::Grid) = g.l / g.n
@inline wavenumber_int(g::Grid, i) = i - 1 - ifelse(i <= (g.n + 1) >> 1, 0, g.n)

struct Field{D,A} <: AbstractArray{Float64,D}
    grid::Grid{D}
    data::A
    function Field(g::Grid, b::Backend)
        data = KernelAbstractions.zeros(b, Float64, ntuple(Returns(g.n), dim(g)))
        new{dim(g),typeof(data)}(g, data)
    end
    Field(g::Grid, data) = new{dim(g),typeof(data)}(g, data)
end

vectorfield(g::Grid, b::Backend) = ntuple(i -> Field(g, b), dim(g))
tensorfield(g::Grid{2}, b::Backend) = ntuple(i -> Field(g, b), 3)
tensorfield(g::Grid{3}, b::Backend) = ntuple(i -> Field(g, b), 6)

struct LazyField{F,D,S} <: AbstractArray{Float64,D}
    func::F
    grid::Grid{D}
    stuff::S
    LazyField(func, grid, stuff...) =
        new{typeof(func),dim(grid),typeof(stuff)}(func, grid, stuff)
end

Adapt.adapt_structure(to, u::Field) = Field(u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::LazyField) =
    LazyField(adapt(to, u.func), u.grid, adapt(to, u.stuff)...)

Base.size(u::Field) = ntuple(Returns(u.grid.n), dim(u.grid))
Base.size(u::LazyField) = ntuple(Returns(u.grid.n), dim(u.grid))

@inline Base.getindex(u::Field{2}, I::Vararg{Int,2}) =
    getindex(u.data, map(i -> mod1(i, u.grid.n), I)...)
@inline Base.getindex(u::Field{3}, I::Vararg{Int,3}) =
    getindex(u.data, map(i -> mod1(i, u.grid.n), I)...)

@inline Base.getindex(u::LazyField{F,2}, I::Vararg{Int,2}) where {F} =
    u.func(u.stuff..., CartesianIndex(I))
@inline Base.getindex(u::LazyField{F,3}, I::Vararg{Int,3}) where {F} =
    u.func(u.stuff..., CartesianIndex(I))

@inline Base.setindex!(u::Field{2}, val, I::Vararg{Int,2}) =
    setindex!(u.data, val, map(i -> mod1(i, u.grid.n), I)...)
@inline Base.setindex!(u::Field{3}, val, I::Vararg{Int,3}) =
    setindex!(u.data, val, map(i -> mod1(i, u.grid.n), I)...)

@inline right(I::CartesianIndex, i, r) =
    CartesianIndex(ntuple(j -> ifelse(j == i, I[j] + r, I[j]), length(I.I)))

Base.show(io::IO, u::Field) = print(io, "Field($(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::LazyField) = print(
    io,
    "LazyField(",
    join((u.func, u.grid, map(s -> "::$(typeof(s))", u.stuff)...), ", ")...,
    ")",
)
Base.show(io::IO, ::MIME"text/plain", u::Field) =
    print(io, join(map(string, size(u)), "×")..., " ", u)
Base.show(io::IO, ::MIME"text/plain", u::LazyField) =
    print(io, join(map(string, size(u)), "×")..., " ", u)

stresstensor(u, visc) =
    if dim(u[1].grid) == 2
        stress(u, visc, 1, 1), stress(u, visc, 2, 2), stress(u, visc, 1, 2)
    else
        stress(u, visc, 1, 1),
        stress(u, visc, 2, 2),
        stress(u, visc, 3, 3),
        stress(u, visc, 1, 2),
        stress(u, visc, 2, 3),
        stress(u, visc, 3, 1)
    end

stress(u, visc, i, j) =
    LazyField(u[1].grid, u, visc, i, j) do u, visc, i, j, I
        @inline
        h = spacing(u[1].grid)
        ui_rightj = u[i][right(I, j, i != j)]
        ui_leftj = u[i][right(I, j, (i != j) - 1)]
        uj_righti = u[j][right(I, i, j != i)]
        uj_lefti = u[j][right(I, i, (j != i) - 1)]
        etaj_ui = (ui_rightj + ui_leftj) / 2
        etai_uj = (uj_righti + uj_lefti) / 2
        dj_ui = (ui_rightj - ui_leftj) / h
        di_uj = (uj_righti - uj_lefti) / h
        etaj_ui * etai_uj - visc * (dj_ui + di_uj)
    end

tensordivergence!(du, σ, doadd) =
# AK.foreachindex(du[1].data) do ilin
    myforeachindex(du[1].data) do ilin
        @inline
        g = du[1].grid
        D = dim(g)
        h = spacing(g)
        I = linear2cartesian(g, ilin)
        if D == 2
            σ11, σ22, σ12 = σ
            du1, du2 = du
            I1, I2 = I.I
            du1[I] =
                doadd * du1[I] - (σ11[I1+1, I2] - σ11[I1, I2]) / h -
                (σ12[I1, I2] - σ12[I1, I2-1]) / h
            du2[I] =
                doadd * du2[I] - (σ12[I1, I2] - σ12[I1-1, I2]) / h -
                (σ22[I1, I2+1] - σ22[I1, I2]) / h
        else
            σ11, σ22, σ33, σ12, σ23, σ31 = σ
            du1, du2, du3 = du
            I1, I2, I3 = I.I
            du1[I] =
                doadd * du1[I] - (σ11[I1+1, I2, I3] - σ11[I1, I2, I3]) / h -
                (σ12[I1, I2+1, I3] - σ12[I1, I2, I3]) / h -
                (σ31[I1, I2, I3+1] - σ31[I1, I2, I3]) / h
            du2[I] =
                doadd * du2[I] - (σ12[I1+1, I2, I3] - σ12[I1, I2, I3]) / h -
                (σ22[I1, I2+1, I3] - σ22[I1, I2, I3]) / h -
                (σ23[I1, I2, I3+1] - σ23[I1, I2, I3]) / h
            du3[I] =
                doadd * du3[I] - (σ31[I1+1, I2, I3] - σ31[I1, I2, I3]) / h -
                (σ23[I1, I2+1, I3] - σ23[I1, I2, I3]) / h -
                (σ33[I1, I2, I3+1] - σ33[I1, I2, I3]) / h
        end
    end

tensordivergence_nonsym!(du, σ, doadd) =
# AK.foreachindex(du[1].data) do ilin
    myforeachindex(du[1].data) do ilin
        @inline
        g = du[1].grid
        D = dim(g)
        h = spacing(g)
        I = linear2cartesian(g, ilin)
        if D == 2
            σ11, σ21, σ12, σ22 = σ
            du1, du2 = du
            I1, I2 = I.I
            du1[I] =
                doadd * du1[I] - #
                (σ11[I1+1, I2] - σ11[I1, I2]) / h - #
                (σ12[I1, I2] - σ12[I1, I2-1]) / h
            du2[I] =
                doadd * du2[I] - #
                (σ21[I1, I2] - σ21[I1-1, I2]) / h - #
                (σ22[I1, I2+1] - σ22[I1, I2]) / h
        else
            σ11, σ21, σ31, σ12, σ22, σ32, σ13, σ23, σ33 = σ
            du1, du2, du3 = du
            I1, I2, I3 = I.I
            du1[I] =
                doadd * du1[I] - #
                (σ11[I1+1, I2, I3] - σ11[I1, I2, I3]) / h -
                (σ12[I1, I2+1, I3] - σ12[I1, I2, I3]) / h -
                (σ13[I1, I2, I3+1] - σ13[I1, I2, I3]) / h
            du2[I] =
                doadd * du2[I] - #
                (σ21[I1+1, I2, I3] - σ21[I1, I2, I3]) / h -
                (σ22[I1, I2+1, I3] - σ22[I1, I2, I3]) / h -
                (σ23[I1, I2, I3+1] - σ23[I1, I2, I3]) / h
            du3[I] =
                doadd * du3[I] - #
                (σ31[I1+1, I2, I3] - σ31[I1, I2, I3]) / h -
                (σ32[I1, I2+1, I3] - σ32[I1, I2, I3]) / h -
                (σ33[I1, I2, I3+1] - σ33[I1, I2, I3]) / h
        end
    end

function poissonsolver(grid, backend)
    (; l, n) = grid

    D = dim(grid)
    h = spacing(grid)

    # Since we use rfft, the first dimension is halved
    kmax = if D == 2
        div(n, 2) + 1, n
    else
        div(n, 2) + 1, n, n
    end

    # Placeholders for intermediate results
    phat = KernelAbstractions.allocate(backend, ComplexF64, kmax)
    p = KernelAbstractions.allocate(backend, Float64, ntuple(Returns(n), D))
    plan = plan_rfft(p)

    (; plan, phat)
end

divergence!(d, u) =
    AK.foreachindex(d.data) do ilin
        @inline
        g = d.grid
        D = dim(g)
        h = spacing(g)
        I = linear2cartesian(g, ilin)
        if D == 2
            d[I] =
                (u[1][I[1], I[2]] - u[1][I[1]-1, I[2]]) / h +
                (u[2][I[1], I[2]] - u[2][I[1], I[2]-1]) / h
        elseif D == 3
            d[I] =
                (u[1][I[1], I[2], I[3]] - u[1][I[1]-1, I[2], I[3]]) / h +
                (u[2][I[1], I[2], I[3]] - u[2][I[1], I[2]-1, I[3]]) / h +
                (u[3][I[1], I[2], I[3]] - u[3][I[1], I[2], I[3]-1]) / h
        end
    end

pressuregradient!(u, p) =
# AK.foreachindex(p.data) do ilin
    myforeachindex(p.data) do ilin
        @inline
        g = p.grid
        D = dim(g)
        h = spacing(g)
        I = linear2cartesian(g, ilin)
        if D == 2
            u[1][I] -= (p[I[1]+1, I[2]] - p[I[1], I[2]]) / h
            u[2][I] -= (p[I[1], I[2]+1] - p[I[1], I[2]]) / h
        else
            u[1][I] -= (p[I[1]+1, I[2], I[3]] - p[I[1], I[2], I[3]]) / h
            u[2][I] -= (p[I[1], I[2]+1, I[3]] - p[I[1], I[2], I[3]]) / h
            u[3][I] -= (p[I[1], I[2], I[3]+1] - p[I[1], I[2], I[3]]) / h
        end
    end

function project!(u, p, poisson)
    (; plan, phat) = poisson
    g = p.grid
    D = dim(g)

    # Divergence of tentative velocity field
    divergence!(p, u)

    # Solve the Poisson equation

    # Fourier transform of right hand side
    mul!(phat, plan, p.data)

    # Solve for coefficients in Fourier space
    if D == 2
        # @. ahat = 4 / dx(grid)^2 * sinpi(k / n)^2
        # AK.foreachindex(phat) do ilin
        myforeachindex(phat) do ilin
            I = CartesianIndices(size(phat))[ilin]
            h = spacing(g)
            k1 = I[1] - 1 # RFFT wavenumber
            k2 = wavenumber_int(g, I[2]) # FFT wavenumber
            denom = -4 / h^2 * (sinpi(k1 / g.n)^2 + sinpi(k2 / g.n)^2)
            phat[I] /= denom
        end
    else
        # AK.foreachindex(phat) do ilin
        myforeachindex(phat) do ilin
            I = CartesianIndices(size(phat))[ilin]
            h = spacing(g)
            k1 = I[1] - 1 # RFFT wavenumber
            k2 = wavenumber_int(g, I[2]) # FFT wavenumber
            k3 = wavenumber_int(g, I[3]) # FFT wavenumber
            denom = -4 / h^2 * (sinpi(k1 / g.n)^2 + sinpi(k2 / g.n)^2 + sinpi(k3 / g.n)^2) # Finite difference Laplacian in Fourier space
            phat[I] /= denom
        end
    end

    # Pressure is determined up to constant. We set this to 0 (instead of
    # phat[1] / 0 = Inf)
    # Note use of singleton range 1:1 instead of scalar index 1
    # (otherwise CUDA gets annoyed)
    phat[1:1] .= 0

    # Inverse Fourier transform
    ldiv!(p.data, plan, phat)

    # Apply pressure correction term
    pressuregradient!(u, p)

    u
end

function step_forwardeuler!(u, du, p, poisson, visc, dt)
    σ = stresstensor(u, visc)
    tensordivergence!(du, σ, false)
    # AK.foreachindex(u[1].data) do i
    myforeachindex(u[1].data) do i
        for (u, du) in zip(u, du)
            u[i] += dt * du[i]
        end
    end
    project!(u, p, poisson)
end

function step_wray3!(u, du, u0, p, poisson, visc, dt)
    a = 8 / 15, 5 / 12, 3 / 4
    b = 1 / 4, 0 / 1
    c = 0 / 1, 8 / 15, 2 / 3
    nstage = length(a)

    for (u, u0) in zip(u, u0)
        copyto!(u0.data, u.data)
    end

    for i = 1:nstage
        σ = stresstensor(u, visc)
        tensordivergence!(du, σ, false)

        # Compute u = project(u0 + dt * a[i] * du)
        for (du, u, u0) in zip(du, u, u0)
            i == 1 || copyto!(u.data, u0.data) # Skip first iter
            axpy!(a[i] * dt, du.data, u.data)
        end
        project!(u, p, poisson)

        # Compute u0 = u0 + dt * b[i] * du
        i == nstage || for (du, u0) in zip(du, u0)
            axpy!(b[i] * dt, du.data, u0.data) # Skip last iter
        end
    end
end

function propose_timestep(u, visc, cfl)
    h = spacing(u[1].grid)
    umax = maximum(u -> maximum(abs, u.data), u)
    dt_adv = cfl * h / umax
    dt_visc = cfl * h^2 / visc
    min(dt_adv, dt_visc)
end

vorticity!(w, u) =
# AK.foreachindex(w.data) do ilin
    myforeachindex(w.data) do ilin
        I = linear2cartesian(w.grid, ilin)
        h = spacing(w.grid)
        w[I] = -(u[1][I[1], I[2]+1] - u[1][I[1], I[2]]) / h
        +(u[2][I[1]+1, I[2]] - u[2][I[1], I[2]]) / h
    end

peak_profile(k; kpeak) = k^4 * exp(-2 * (k / kpeak)^2)

@inline function get_wavenumbers(grid)
    (; n) = grid
    kmax = div(n, 2)
    if dim(grid) == 2
        kmax + 1, n
    else
        kmax + 1, n, n
    end
end

applymask!(grid, k, Emask, mask, E) =
    AK.foreachindex(mask) do ilin
        D = dim(grid)
        wavenumbers = get_wavenumbers(grid)
        I = CartesianIndices(wavenumbers)[ilin]
        kk = if D == 2
            k1 = I[1] - 1
            k2 = wavenumber_int(grid, I[2])
            k1^2 + k2^2
        else
            k1 = I[1] - 1
            k2 = wavenumber_int(grid, I[2])
            k3 = wavenumber_int(grid, I[3])
            k1^2 + k2^2 + k3^2
        end
        m = k^2 ≤ kk < (k + 1)^2
        mask[I] = m
        Emask[I] = m * E[I]
    end

function randomfield(profile, grid, backend, poisson; totalenergy = 1, rng, kwargs...)
    (; n) = grid
    (; plan) = poisson
    D = dim(grid)

    # Create random field and make it divergence free
    u = vectorfield(grid, backend)
    p = Field(grid, backend)
    foreach(u -> randn!(rng, u.data), u)
    project!(u, p, poisson)

    uhat = map(u -> plan * u.data, u)
    for uhat in uhat
        uhat ./= n^D # FFT factor
    end

    # RFFT exploits conjugate symmetry, so we only need half the modes
    kmax = div(n, 2)
    wavenumbers = if dim(grid) == 2
        kmax + 1, n
    else
        kmax + 1, n, n
    end

    # Allocate arrays
    E = similar(uhat[1], Float64)
    Emask = similar(E)
    mask = similar(E, Bool)

    # Compute energy
    if dim(grid) == 2
        @. E = (abs2(uhat[1]) + abs2(uhat[2])) / 2
    else
        @. E = (abs2(uhat[1]) + abs2(uhat[2]) + abs2(uhat[3])) / 2
    end

    # Maximum partially resolved wavenumber (sqrt(dim) * kmax)
    kdiag = floor(Int, sqrt(3) * kmax)

    # Sum of shell weights
    totalprofile = sum(k -> profile(k; kwargs...), 0:kdiag)

    # Adjust energy in each partially resolved shell [k, k+1)
    for k = 0:kdiag
        applymask!(grid, k, Emask, mask, E)
        Eshell = sum(Emask) + sum(selectdim(Emask, 1, 2:kmax)) # Current energy in shell
        E0 = totalenergy * profile(k; kwargs...) / totalprofile # Desired energy in shell
        factor = sqrt(E0 / Eshell) # E = u^2 / 2
        for uhat in uhat
            @. uhat = ifelse(mask, factor * uhat, uhat)
        end
    end

    # @show sum(uhat) do uhat
    #     EE = abs2.(uhat) / 2
    #     sum(EE) + sum(selectdim(EE, 1, 2:kmax)) # Current energy in shell
    # end

    # Inverse RFFT
    for (u, uhat) in zip(u, uhat)
        ldiv!(u.data, plan, uhat)
        u.data .*= n^D # FFT factor
    end

    # Normally, adjusting the amplitude of the 3D vector uhat(k)
    # does not remove divergence freeness, since the divergence becomes
    # i k dot uhat(k) in  Fourier space, which stays zero if uhat is scaled.
    # But since our divergence is discrete, defined through staggered finite
    # differences, this is no longer exactly the case. So we project again
    # to correct for this (minor?) error.
    project!(u, p, poisson)

    # The velocity now has
    # the correct spectrum,
    # random phase shifts,
    # random orientations,
    # and is also divergence free.
    u
end

function getshells(grid, shells)
    (; n) = grid
    kmax = div(n, 2)
    D = dim(grid)

    spectralrange = if D == 2
        kmax + 1, n
    else
        kmax + 1, n, n
    end

    # Get squared wavenumbers in an RFFT-shaped array
    kk = map(CartesianIndices(spectralrange)) do I
        if D == 2
            k1 = I[1] - 1
            k2 = wavenumber_int(grid, I[2])
            k1^2 + k2^2
        else
            k1 = I[1] - 1
            k2 = wavenumber_int(grid, I[2])
            k3 = wavenumber_int(grid, I[3])
            k1^2 + k2^2 + k3^2
        end
    end

    # Flatten since we work with linear RFFT indices
    kk = reshape(kk, :)

    isort = sortperm(kk) # Permutation for sorting the wavenumbers
    kksort = kk[isort]

    # Get linear RFFT indices and corresponding waveumbers for each shell
    map(shells) do i
        # Since the wavenumbers are sorted, we just need to find the start and stop of each shell.
        # The linear indices for that shell is then given by the permutation in that range.
        jstart = findfirst(≥(i^2), kksort)
        jstop = findfirst(≥((i + 1)^2), kksort)
        isnothing(jstop) && (jstop = length(kksort) + 1) # findfirst may return nothing
        jstop -= 1
        inds = isort[jstart:jstop] # Linear indices of the i-th shell

        # We need to adapt the shells for RFFT.
        # Consider the following example:
        #
        # julia> n, kmax = 8, 4;
        # julia> u = randn(n, n, n);
        # julia> f = fft(u); r = rfft(u);
        # julia> sum(abs2, f)
        # 275142.33506202063
        # julia> sum(abs2, r) + sum(abs2, selectdim(r, 1, 2:kmax))
        # 275142.3350620207
        #
        # To compute the energy of the FFT, we need an additional term for RFFT.
        # The second term sums over all the x-indices except for 1 and kmax + 1.
        # We thus need to add indices to account for the conjugate symmetry in RFFT.
        # For an RFFT array r of size (kmax + 1, n, n), we have the linear index relation
        # r[i] == r[x, y, z]
        # if
        # i == x + (y - 1) * (kmax + 1) + (z - 1) * (kmax + 1) * n.
        # We therefore need to exclude the indices:
        # (x == 1), i.e. (i % (kmax + 1) == 1), and
        # (x == kmax + 1), i.e. (i % (kmax + 1) == 0).
        # We only keep i if (i % (kmax + 1) > 1).
        conjinds = filter(j -> j % (kmax + 1) > 1, inds)

        (; shell = i, inds = (inds, conjinds), kk = (kk[inds], kk[conjinds]))
    end
end

function spectral_stuff(grid, backend; npoint = nothing)
    (; l, n) = grid
    T = typeof(l)
    D = dim(grid)

    kmax = div(n, 2)

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    kcut = div(2 * kmax, 3)
    # kcut = kmax
    if isnothing(npoint)
        kuse = 1:kcut
    else
        kuse = logrange(T(1), T(kcut), npoint)
        kuse = sort(unique(round.(Int, kuse)))
    end

    shells = getshells(grid, kuse)
    inds = map(s -> vcat(s.inds...), shells) # Include conjugate indices

    # Put indices on GPU
    inds = map(adapt(backend), inds)

    specsize = D == 2 ? (kmax + 1, n) : (kmax + 1, n, n)
    ehat = KernelAbstractions.allocate(backend, Float64, specsize)

    (; shells = inds, k = 2π / l * kuse, ehat)
end

function spectrum(u, stuff, poisson)
    (; shells, k, ehat) = stuff
    (; plan, phat) = poisson
    g = u[1].grid
    fact = g.n^dim(g)
    fill!(ehat, 0)
    for u in u
        mul!(phat, plan, u.data)
        @. ehat += abs2(phat / fact) / 2
    end
    s = map(shells) do shell
        sum(view(ehat, shell))
    end
    (; k, s)
end

function tophat(grid, compression)
    @assert isodd(compression)
    r = div(compression, 2)
    w = fill(1 / compression, compression)
    w
end

function gaussian(grid, Δ, nσ)
    h = spacing(grid)
    σ = Δ / sqrt(12)
    r = round(Int, nσ * σ / h)
    x = h * ((-r):r)
    w = @. exp(-x^2 / 2σ^2)
    w ./= sum(w) # Normalize
    w
end

function composekernel(F, G)
    f = F
    g = G
    I = div(length(f), 2)
    J = div(length(g), 2)
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
    w_double
end

function kernelproduct(::Grid{2}, kernels)
    w1 = reshape(kernels[1], :)
    w2 = reshape(kernels[2], 1, :)
    w1 .* w2
end
function kernelproduct(::Grid{3}, kernels)
    w1 = reshape(kernels[1], :)
    w2 = reshape(kernels[2], 1, :)
    w3 = reshape(kernels[3], 1, 1, :)
    w1 .* w2 .* w3
end

function applyfilter!(ubar, u, kernel)
    W = kernel
    R = map(n -> div(n, 2), size(W)) |> CartesianIndex
    RR = (-R):R
    grid = ubar.grid
    # AK.foreachindex(ubar.data) do ilin
    myforeachindex(ubar.data) do ilin
        @inline
        I = linear2cartesian(grid, ilin)
        s = zero(eltype(ubar))
        for (r, w) in zip(RR, W)
            s += w * u[I+r]
        end
        ubar[I] = s
    end
end

function lazyfilter(u, kernel)
    W = kernel
    R = map(n -> div(n, 2), size(W)) |> CartesianIndex
    RR = (-R):R
    grid = u.grid
    LazyField(grid, u, W, RR) do u, W, RR, I
        @inline
        s = zero(eltype(u))
        for (r, w) in zip(RR, W)
            s += w * u[I+r]
        end
        s
    end
end

function coarsegrain!(ubar, u, stag)
    g = u.grid
    gbar = ubar.grid
    factor = div(g.n, gbar.n)
    a = div(factor, 2)
    @assert factor * gbar.n == g.n
    # AK.foreachindex(ubar.data) do ilin
    myforeachindex(ubar.data) do ilin
        @inline
        Ibar = linear2cartesian(gbar, ilin)
        I = ntuple(dim(g)) do i
            factor * Ibar[i] - !(stag[i]) * a
        end |> CartesianIndex
        ubar[Ibar] = u[I]
    end
end

function lazycoarsegrain(gbar::Grid, u, stag)
    g = u.grid
    factor = div(g.n, gbar.n)
    a = div(factor, 2)
    @assert factor * gbar.n == g.n
    LazyField(gbar, u, stag, factor, a) do u, stag, factor, a, Ibar
        @inline
        I = ntuple(dim(g)) do i
            factor * Ibar[i] - !(stag[i]) * a
        end |> CartesianIndex
        u[I]
    end
end

getsetup() = (;
    D = Val(2),
    l = 2π,
    # n_dns = 2700, n_les = 300, visc = 5e-4,
    n_dns = 300,
    n_les = 100,
    visc = 1e-3,
    Δ_ratio = 2,
    nσ = 3, # Number of sigmas out in gaussian support
)

function dnsaid(setup)
    (; n_dns, n_les, visc, l, D) = setup
    tstop = 0.005
    cfl = 0.45
    profile, args = k -> (k > 0) * k^-3.0, (; totalenergy = 1.0)
    # profile, args = peak_profile, (; totalenergy = 1.0, kpeak = 5)
    g_dns = Grid(D, l, n_dns)
    g_les = Grid(D, l, n_les)
    backend = KernelAbstractions.CPU()

    # Allocate fields
    poisson_dns = poissonsolver(g_dns, backend)
    poisson_les = poissonsolver(g_les, backend)
    p_dns = Field(g_dns, backend)
    p_dns2 = Field(g_dns, backend)
    p_les = Field(g_les, backend)
    u_dns = randomfield(profile, g_dns, backend, poisson_dns; rng = Xoshiro(0), args...)
    du_dns = vectorfield(g_dns, backend)
    du_dns2 = vectorfield(g_dns, backend)
    du_les = vectorfield(g_les, backend)
    u_dnsΔH_concrete = vectorfield(g_les, backend)
    uΔHπ = vectorfield(g_les, backend)
    u_nomo = vectorfield(g_les, backend)
    u_c = vectorfield(g_les, backend)
    u_cf = vectorfield(g_les, backend)
    u_cfd = vectorfield(g_les, backend)
    u_cfd_symm = vectorfield(g_les, backend)

    # Filter kernels
    comp = div(n_dns, n_les)
    h = spacing(g_dns)
    H = spacing(g_les)
    FH = tophat(g_dns, comp)
    FΔ = gaussian(g_dns, 2H, 3)
    FΔH = composekernel(FH, FΔ)
    GΔH = kernelproduct(g_dns, ntuple(Returns(FΔH), D))
    GΔHi = ntuple(i -> kernelproduct(g_dns, ntuple(j -> i == j ? FΔ : FΔH, D)), D)

    # Position indicators for staggered grid
    faces = ntuple(i -> ntuple(==(i), D), D)
    center = ntuple(Returns(false), D)
    corner2D = (true, true)
    corners3D = (false, true, true), (true, false, true), (true, true, false)

    # Initialize LES fields
    u_dnsΔH = map(1:D) do i
        a = lazyfilter(u_dns[i], GΔH)
        b = lazycoarsegrain(g_les, a, faces[i])
        b
    end
    for i = 1:D
        # AK.foreachindex(u_nomo[i].data) do ilin
        myforeachindex(u_nomo[i].data) do ilin
            @inline
            u_dnsΔH_concrete[i][ilin] = u_dnsΔH[i][ilin]
        end
        copyto!(u_nomo[i].data, u_dnsΔH_concrete[i].data)
        copyto!(u_c[i].data, u_dnsΔH_concrete[i].data)
        copyto!(u_cf[i].data, u_dnsΔH_concrete[i].data)
        copyto!(u_cfd[i].data, u_dnsΔH_concrete[i].data)
        copyto!(u_cfd_symm[i].data, u_dnsΔH_concrete[i].data)
    end

    t = 0.0
    itime = 0
    while t < tstop
        dt = propose_timestep(u_dns, visc, cfl)
        dt = min(dt, tstop - t)
        t += dt
        itime += 1
        @show t

        # DNS stuff
        σ_dns = stresstensor(u_dns, visc)
        tensordivergence!(du_dns, σ_dns, false)
        project!(du_dns, p_dns, poisson_dns)
        r = if D == 2
            r11 = LazyField((σ_dns, p_dns, I) -> σ_dns[1][I] + p_dns[I], g_dns, σ_dns, p_dns)
            r22 = LazyField(
                (σ_dns, p_dns, I) -> σ_dns[2][I] + p_dns[I],
                g_dns,
                σ_dns,
                p_dns,
            )
            r12 = σ_dns[3]
            r11, r22, r12
        else
            r11 = LazyField((σ_dns, p_dns, I) -> σ_dns[1][I] + p_dns[I], g_dns, σ_dns, p_dns)
            r22 = LazyField(
                (σ_dns, p_dns, I) -> σ_dns[2][I] + p_dns[I],
                g_dns,
                σ_dns,
                p_dns,
            )
            r33 = LazyField(
                (σ_dns, p_dns, I) -> σ_dns[3][I] + p_dns[I],
                g_dns,
                σ_dns,
                p_dns,
            )
            r12 = σ_dns[4]
            r23 = σ_dns[5]
            r31 = σ_dns[6]
            r11, r22, r33, r12, r23, r31
        end

        rΔHi = if D == 2
            r11, r22, r12 = r
            r11_bar = lazyfilter(r11, GΔHi[1])
            r21_bar = lazyfilter(r12, GΔHi[1])
            r12_bar = lazyfilter(r12, GΔHi[2])
            r22_bar = lazyfilter(r22, GΔHi[2])
            r11_c = lazycoarsegrain(g_les, r11_bar, center)
            r21_c = lazycoarsegrain(g_les, r21_bar, corner2D)
            r12_c = lazycoarsegrain(g_les, r12_bar, corner2D)
            r22_c = lazycoarsegrain(g_les, r22_bar, center)
            r11_c, r21_c, r12_c, r22_c
        else
            r11, r22, r33, r12, r23, r31 = r
            r11_bar = lazyfilter(r11, GΔHi[1])
            r21_bar = lazyfilter(r12, GΔHi[1])
            r31_bar = lazyfilter(r31, GΔHi[1])
            r12_bar = lazyfilter(r12, GΔHi[2])
            r22_bar = lazyfilter(r22, GΔHi[2])
            r32_bar = lazyfilter(r23, GΔHi[2])
            r13_bar = lazyfilter(r31, GΔHi[3])
            r23_bar = lazyfilter(r23, GΔHi[3])
            r33_bar = lazyfilter(r33, GΔHi[3])
            r11_c = lazycoarsegrain(g_les, r11_bar, center)
            r21_c = lazycoarsegrain(g_les, r21_bar, corners3D[3])
            r31_c = lazycoarsegrain(g_les, r31_bar, corners3D[2])
            r12_c = lazycoarsegrain(g_les, r12_bar, corners3D[3])
            r22_c = lazycoarsegrain(g_les, r22_bar, center)
            r32_c = lazycoarsegrain(g_les, r32_bar, corners3D[1])
            r13_c = lazycoarsegrain(g_les, r13_bar, corners3D[2])
            r23_c = lazycoarsegrain(g_les, r23_bar, corners3D[1])
            r33_c = lazycoarsegrain(g_les, r33_bar, center)
            r11_c, r21_c, r31_c, r12_c, r22_c, r32_c, r13_c, r23_c, r33_c
        end

        rΔHi_symm = if D == 2
            rΔHi_11, rΔHi_21, rΔHi_12, rΔHi_22 = rΔHi
            rΔHi_symm_12 = LazyField(
                (r21, r12, I) -> (r21[I] + r12[I]) / 2,
                g_les,
                rΔHi_21,
                rΔHi_12,
            )
            rΔHi_11, rΔHi_22, rΔHi_symm_12
        else
            rΔHi_11, rΔHi_21, rΔHi_31, rΔHi_12, rΔHi_22, rΔHi_32, rΔHi_13, rΔHi_23, rΔHi_33 = rΔHi
            rΔHi_symm_12 = LazyField(
                (r21, r12, I) -> (r21[I] + r12[I]) / 2,
                g_les,
                rΔHi_21,
                rΔHi_12,
            )
            rΔHi_symm_23 = LazyField(
                (r32, r23, I) -> (r32[I] + r23[I]) / 2,
                g_les,
                rΔHi_32,
                rΔHi_23,
            )
            rΔHi_symm_31 = LazyField(
                (r31, r13, I) -> (r31[I] + r13[I]) / 2,
                g_les,
                rΔHi_31,
                rΔHi_13,
            )
            rΔHi_11, rΔHi_22, rΔHi_33, rΔHi_symm_12, rΔHi_symm_23, rΔHi_symm_31
        end

        rΔH = if D == 2
            r11, r22, r12 = r
            r11_bar = lazyfilter(r11, GΔH)
            r22_bar = lazyfilter(r22, GΔH)
            r12_bar = lazyfilter(r12, GΔH)
            r11_c = lazycoarsegrain(g_les, r11_bar, center)
            r22_c = lazycoarsegrain(g_les, r22_bar, center)
            r12_c = lazycoarsegrain(g_les, r12_bar, corner2D)
            r11_c, r22_c, r12_c
        else
            r11, r22, r33, r12, r23, r31 = r
            r11_bar = lazyfilter(r11, GΔH)
            r22_bar = lazyfilter(r22, GΔH)
            r33_bar = lazyfilter(r33, GΔH)
            r12_bar = lazyfilter(r12, GΔH)
            r23_bar = lazyfilter(r23, GΔH)
            r31_bar = lazyfilter(r31, GΔH)
            r11_c = lazycoarsegrain(g_les, r11_bar, center)
            r22_c = lazycoarsegrain(g_les, r22_bar, center)
            r33_c = lazycoarsegrain(g_les, r33_bar, center)
            r12_c = lazycoarsegrain(g_les, r12_bar, corners3D[3])
            r23_c = lazycoarsegrain(g_les, r23_bar, corners3D[1])
            r31_c = lazycoarsegrain(g_les, r31_bar, corners3D[2])
            r11_c, r22_c, r33_c, r12_c, r23_c, r31_c
        end

        u_dnsΔH = map(u -> lazyfilter(u, GΔH), u_dns)
        u_dnsΔH_coarse = ntuple(i -> lazycoarsegrain(g_les, u_dnsΔH[i], faces[i]), D)

        # No-model
        σ_nomo = stresstensor(u_nomo, visc)
        tensordivergence!(du_les, σ_nomo, false)
        project!(du_les, p_les, poisson_les)
        foreach(i -> axpy!(dt, du_les[i].data, u_nomo[i].data), 1:D)

        # Classic
        σ_dns2 = stresstensor(u_dnsΔH, visc)
        tensordivergence!(du_dns2, σ_dns2, false)
        project!(du_dns2, p_dns2, poisson_dns)
        r2 = if D == 2
            σ2_11, σ2_22, σ2_12 = σ_dns2
            r2_11 = LazyField(
                (σ2_11, p_dns2, I) -> σ2_11[I] + p_dns2[I],
                g_dns,
                σ2_11,
                p_dns2,
            )
            r2_22 = LazyField(
                (σ2_22, p_dns2, I) -> σ2_22[I] + p_dns2[I],
                g_dns,
                σ2_22,
                p_dns2,
            )
            r2_12 = σ2_12
            r2_11_c = lazycoarsegrain(g_les, r2_11, center)
            r2_22_c = lazycoarsegrain(g_les, r2_22, center)
            r2_12_c = lazycoarsegrain(g_les, r2_12, corner2D)
            r2_11_c, r2_22_c, r2_12_c
        else
            σ2_11, σ2_22, σ2_33, σ2_12, σ2_23, σ2_31 = σ_dns2
            r2_11 = LazyField(
                (σ2_11, p_dns2, I) -> σ2_11[I] + p_dns2[I],
                g_dns,
                σ2_11,
                p_dns2,
            )
            r2_22 = LazyField(
                (σ2_22, p_dns2, I) -> σ2_22[I] + p_dns2[I],
                g_dns,
                σ2_22,
                p_dns2,
            )
            r2_33 = LazyField(
                (σ2_33, p_dns2, I) -> σ2_33[I] + p_dns2[I],
                g_dns,
                σ2_33,
                p_dns2,
            )
            r2_12 = σ2_12
            r2_23 = σ2_23
            r2_31 = σ2_31
            r2_11_c = lazycoarsegrain(g_les, r2_11, center)
            r2_22_c = lazycoarsegrain(g_les, r2_22, center)
            r2_33_c = lazycoarsegrain(g_les, r2_33, center)
            r2_12_c = lazycoarsegrain(g_les, r2_12, corners3D[3])
            r2_23_c = lazycoarsegrain(g_les, r2_23, corners3D[1])
            r2_31_c = lazycoarsegrain(g_les, r2_31, corners3D[2])
            r2_11_c, r2_22_c, r2_33_c, r2_12_c, r2_23_c, r2_31_c
        end
        σ_c = stresstensor(u_c, visc)
        tensordivergence!(du_les, σ_c, false) # Overwrite du
        project!(du_les, p_les, poisson_les) # σ_both is projected
        r_both =
            map((r1, r2) -> LazyField((r1, r2, I) -> r1[I] - r2[I], g_les, r1, r2), rΔH, r2)
        tensordivergence!(du_les, r_both, true) # Add to existing du, don't project
        foreach(i -> axpy!(dt, du_les[i].data, u_c[i].data), 1:D)

        # Classic+Flux
        σ2 = stresstensor(u_dnsΔH_coarse, visc)
        σ_cf = stresstensor(u_cf, visc)
        σ_both = map(
            (σ_cf, σ2) -> LazyField((σ_cf, σ2, I) -> σ_cf[I] - σ2[I], g_les, σ_cf, σ2),
            σ_cf,
            σ2,
        )
        tensordivergence!(du_les, σ_both, false) # Overwrite du
        project!(du_les, p_les, poisson_les) # σ_both is projected
        tensordivergence!(du_les, rΔH, true) # Add to existing du, don't project
        foreach(i -> axpy!(dt, du_les[i].data, u_cf[i].data), 1:D)

        # Classic+Flux+Div
        σ2 = stresstensor(u_dnsΔH_coarse, visc)
        σ_cfd = stresstensor(u_cfd, visc)
        σ_both = map(
            (σ_cfd, σ2) ->
                LazyField((σ_cfd, σ2, I) -> σ_cfd[I] - σ2[I], g_les, σ_cfd, σ2),
            σ_cfd,
            σ2,
        )
        tensordivergence!(du_les, σ_both, false) # Overwrite du
        project!(du_les, p_les, poisson_les) # σ_both is projected
        tensordivergence_nonsym!(du_les, rΔHi, true) # Add to existing du, don't project
        foreach(i -> axpy!(dt, du_les[i].data, u_cfd[i].data), 1:D)

        # Classic+Flux+Div symmetrized
        σ2 = stresstensor(u_dnsΔH_coarse, visc)
        σ_cfd_symm = stresstensor(u_cfd_symm, visc)
        σ_both = map(
            (σ1, σ2) -> LazyField((σ1, σ2, I) -> σ1[I] - σ2[I], g_les, σ1, σ2),
            σ_cfd_symm,
            σ2,
        )
        tensordivergence!(du_les, σ_both, false) # Overwrite du
        project!(du_les, p_les, poisson_les) # σ_both is projected
        tensordivergence!(du_les, rΔHi_symm, true) # Add to existing du, don't project
        foreach(i -> axpy!(dt, du_les[i].data, u_cfd_symm[i].data), 1:D)

        # DNS
        foreach(i -> axpy!(dt, du_dns[i].data, u_dns[i].data), 1:D)
    end

    (; u_dns, u_nomo, u_c, u_cf, u_cfd, u_cfd_symm)
end

get_positions(g::Grid) = (;
    faces = ntuple(i -> ntuple(==(i), D), D),
    center = ntuple(Returns(false), D),
    corner2D = (true, true),
    corners3D = ((false, true, true), (true, false, true), (true, true, false)),
)

function dnsaid_project(setup)
    (; l, n_dns, n_les, visc, Δ_ratio, nσ) = setup
    twarm = 0.1
    tstop = 0.01
    cfl = 0.15
    profile, args = k -> (k > 0) * k^-3.0, (; totalenergy = 1.0)
    # profile, args = peak_profile, (; totalenergy = 1.0, kpeak = 5)
    g_dns = Grid(setup.D, l, n_dns)
    g_les = Grid(setup.D, l, n_les)
    D = dim(g_dns)
    backend = defaultbackend()

    # Allocate fields
    poisson_dns = poissonsolver(g_dns, backend)
    poisson_les = poissonsolver(g_les, backend)
    p_dns = Field(g_dns, backend)
    p_les = Field(g_les, backend)
    u_dns = randomfield(profile, g_dns, backend, poisson_dns; rng = Xoshiro(0), args...)
    du_dns = vectorfield(g_dns, backend)
    du_les = vectorfield(g_les, backend)
    u_dnsΔH_concrete = vectorfield(g_les, backend)
    ubar_coarse = vectorfield(g_les, backend)
    u_nomo = vectorfield(g_les, backend)
    u_c = vectorfield(g_les, backend)
    u_cf = vectorfield(g_les, backend)
    u_cfd = vectorfield(g_les, backend)
    u_cfd_symm = vectorfield(g_les, backend)

    # Filter kernels
    comp = div(n_dns, n_les)
    h = spacing(g_dns)
    H = spacing(g_les)
    FH = tophat(g_dns, comp)
    Δ = Δ_ratio * H
    FΔ = gaussian(g_dns, Δ, nσ)
    FΔH = composekernel(FH, FΔ)
    GΔH = kernelproduct(g_dns, ntuple(Returns(FΔH), D))
    GΔHi = ntuple(i -> kernelproduct(g_dns, ntuple(j -> i == j ? FΔ : FΔH, D)), D)

    # Put on device
    GΔH = adapt(backend, GΔH)
    GΔHi = adapt(backend, GΔHi)

    # Position indicators for staggered grid
    (; faces, center, corner2D, corners3D) = get_positions(g_dns)

    # Warmup
    t = 0.0
    itime = 0
    while t < twarm
        dt = propose_timestep(u_dns, visc, cfl)
        dt = min(dt, twarm - t)
        t += dt
        itime += 1

        @info "t = $(round(t; sigdigits=4)) / $(round(twarm; sigdigits=4))"

        # DNS stuff
        step_forwardeuler!(u_dns, du_dns, p_dns, poisson_dns, visc, dt)
    end

    # Initialize LES fields
    for i = 1:D
        ubari = lazyfilter(u_dns[i], GΔH)
        coarsegrain!(ubar_coarse[i], ubari, faces[i])
    end
    project!(ubar_coarse, p_les, poisson_les)

    for i = 1:D
        copyto!(u_nomo[i].data, ubar_coarse[i].data)
        copyto!(u_c[i].data, ubar_coarse[i].data)
        copyto!(u_cf[i].data, ubar_coarse[i].data)
        copyto!(u_cfd[i].data, ubar_coarse[i].data)
        copyto!(u_cfd_symm[i].data, ubar_coarse[i].data)
    end

    t = 0.0
    itime = 0
    while t < tstop
        dt = propose_timestep(u_dns, visc, cfl)
        dt = min(dt, tstop - t)
        t += dt
        itime += 1
        @show t

        # DNS stuff
        σ_dns = stresstensor(u_dns, visc)
        tensordivergence!(du_dns, σ_dns, false)
        project!(du_dns, p_dns, poisson_dns)
        r = if D == 2
            r11 = LazyField((σ_dns, p_dns, I) -> σ_dns[1][I] + p_dns[I], g_dns, σ_dns, p_dns)
            r22 = LazyField(
                (σ_dns, p_dns, I) -> σ_dns[2][I] + p_dns[I],
                g_dns,
                σ_dns,
                p_dns,
            )
            r12 = σ_dns[3]
            r11, r22, r12
        else
            r11 = LazyField((σ_dns, p_dns, I) -> σ_dns[1][I] + p_dns[I], g_dns, σ_dns, p_dns)
            r22 = LazyField(
                (σ_dns, p_dns, I) -> σ_dns[2][I] + p_dns[I],
                g_dns,
                σ_dns,
                p_dns,
            )
            r33 = LazyField(
                (σ_dns, p_dns, I) -> σ_dns[3][I] + p_dns[I],
                g_dns,
                σ_dns,
                p_dns,
            )
            r12 = σ_dns[4]
            r23 = σ_dns[5]
            r31 = σ_dns[6]
            r11, r22, r33, r12, r23, r31
        end

        rΔHi = if D == 2
            r11, r22, r12 = r
            r11_bar = lazyfilter(r11, GΔHi[1])
            r21_bar = lazyfilter(r12, GΔHi[1])
            r12_bar = lazyfilter(r12, GΔHi[2])
            r22_bar = lazyfilter(r22, GΔHi[2])
            r11_c = lazycoarsegrain(g_les, r11_bar, center)
            r21_c = lazycoarsegrain(g_les, r21_bar, corner2D)
            r12_c = lazycoarsegrain(g_les, r12_bar, corner2D)
            r22_c = lazycoarsegrain(g_les, r22_bar, center)
            r11_c, r21_c, r12_c, r22_c
        else
            r11, r22, r33, r12, r23, r31 = r
            r11_bar = lazyfilter(r11, GΔHi[1])
            r21_bar = lazyfilter(r12, GΔHi[1])
            r31_bar = lazyfilter(r31, GΔHi[1])
            r12_bar = lazyfilter(r12, GΔHi[2])
            r22_bar = lazyfilter(r22, GΔHi[2])
            r32_bar = lazyfilter(r23, GΔHi[2])
            r13_bar = lazyfilter(r31, GΔHi[3])
            r23_bar = lazyfilter(r23, GΔHi[3])
            r33_bar = lazyfilter(r33, GΔHi[3])
            r11_c = lazycoarsegrain(g_les, r11_bar, center)
            r21_c = lazycoarsegrain(g_les, r21_bar, corners3D[3])
            r31_c = lazycoarsegrain(g_les, r31_bar, corners3D[2])
            r12_c = lazycoarsegrain(g_les, r12_bar, corners3D[3])
            r22_c = lazycoarsegrain(g_les, r22_bar, center)
            r32_c = lazycoarsegrain(g_les, r32_bar, corners3D[1])
            r13_c = lazycoarsegrain(g_les, r13_bar, corners3D[2])
            r23_c = lazycoarsegrain(g_les, r23_bar, corners3D[1])
            r33_c = lazycoarsegrain(g_les, r33_bar, center)
            r11_c, r21_c, r31_c, r12_c, r22_c, r32_c, r13_c, r23_c, r33_c
        end

        rΔHi_symm = if D == 2
            rΔHi_11, rΔHi_21, rΔHi_12, rΔHi_22 = rΔHi
            rΔHi_symm_12 = LazyField(
                (r21, r12, I) -> (r21[I] + r12[I]) / 2,
                g_les,
                rΔHi_21,
                rΔHi_12,
            )
            rΔHi_11, rΔHi_22, rΔHi_symm_12
        else
            rΔHi_11, rΔHi_21, rΔHi_31, rΔHi_12, rΔHi_22, rΔHi_32, rΔHi_13, rΔHi_23, rΔHi_33 = rΔHi
            rΔHi_symm_12 = LazyField(
                (r21, r12, I) -> (r21[I] + r12[I]) / 2,
                g_les,
                rΔHi_21,
                rΔHi_12,
            )
            rΔHi_symm_23 = LazyField(
                (r32, r23, I) -> (r32[I] + r23[I]) / 2,
                g_les,
                rΔHi_32,
                rΔHi_23,
            )
            rΔHi_symm_31 = LazyField(
                (r31, r13, I) -> (r31[I] + r13[I]) / 2,
                g_les,
                rΔHi_31,
                rΔHi_13,
            )
            rΔHi_11, rΔHi_22, rΔHi_33, rΔHi_symm_12, rΔHi_symm_23, rΔHi_symm_31
        end

        σΔH = if D == 2
            σ11, σ22, σ12 = σ_dns
            σ11_bar = lazyfilter(σ11, GΔH)
            σ22_bar = lazyfilter(σ22, GΔH)
            σ12_bar = lazyfilter(σ12, GΔH)
            σ11_c = lazycoarsegrain(g_les, σ11_bar, center)
            σ22_c = lazycoarsegrain(g_les, σ22_bar, center)
            σ12_c = lazycoarsegrain(g_les, σ12_bar, corner2D)
            σ11_c, σ22_c, σ12_c
        else
            σ11, σ22, σ33, σ12, σ23, σ31 = σ_dns
            σ11_bar = lazyfilter(σ11, GΔH)
            σ22_bar = lazyfilter(σ22, GΔH)
            σ33_bar = lazyfilter(σ33, GΔH)
            σ12_bar = lazyfilter(σ12, GΔH)
            σ23_bar = lazyfilter(σ23, GΔH)
            σ31_bar = lazyfilter(σ31, GΔH)
            σ11_c = lazycoarsegrain(g_les, σ11_bar, center)
            σ22_c = lazycoarsegrain(g_les, σ22_bar, center)
            σ33_c = lazycoarsegrain(g_les, σ33_bar, center)
            σ12_c = lazycoarsegrain(g_les, σ12_bar, corners3D[3])
            σ23_c = lazycoarsegrain(g_les, σ23_bar, corners3D[1])
            σ31_c = lazycoarsegrain(g_les, σ31_bar, corners3D[2])
            σ11_c, σ22_c, σ33_c, σ12_c, σ23_c, σ31_c
        end

        u_dnsΔH = map(u -> lazyfilter(u, GΔH), u_dns)

        # Filter
        for i = 1:D
            ubari = lazyfilter(u_dns[i], GΔH)
            coarsegrain!(ubar_coarse[i], ubari, faces[i])
        end
        project!(ubar_coarse, p_les, poisson_les)

        # No-model
        σ_nomo = stresstensor(u_nomo, visc)
        tensordivergence!(du_les, σ_nomo, false)
        project!(du_les, p_les, poisson_les)
        foreach(i -> axpy!(dt, du_les[i].data, u_nomo[i].data), 1:D)

        # Classic
        σ_dns2 = stresstensor(u_dnsΔH, visc)
        σ2 = if D == 2
            σ2_11, σ2_22, σ2_12 = σ_dns2
            σ2_11_c = lazycoarsegrain(g_les, σ2_11, center)
            σ2_22_c = lazycoarsegrain(g_les, σ2_22, center)
            σ2_12_c = lazycoarsegrain(g_les, σ2_12, corner2D)
            σ2_11_c, σ2_22_c, σ2_12_c
        else
            σ2_11, σ2_22, σ2_33, σ2_12, σ2_23, σ2_31 = σ_dns2
            σ2_11_c = lazycoarsegrain(g_les, σ2_11, center)
            σ2_22_c = lazycoarsegrain(g_les, σ2_22, center)
            σ2_33_c = lazycoarsegrain(g_les, σ2_33, center)
            σ2_12_c = lazycoarsegrain(g_les, σ2_12, corners3D[3])
            σ2_23_c = lazycoarsegrain(g_les, σ2_23, corners3D[1])
            σ2_31_c = lazycoarsegrain(g_les, σ2_31, corners3D[2])
            σ2_11_c, σ2_22_c, σ2_33_c, σ2_12_c, σ2_23_c, σ2_31_c
        end
        σ_c = stresstensor(u_c, visc)
        σ_all = map(
            (σ_c, σ1, σ2) -> LazyField(
                (σ_c, σ1, σ2, I) -> σ_c[I] + σ1[I] - σ2[I],
                g_les,
                σ_c,
                σ1,
                σ2,
            ),
            σ_c,
            σΔH,
            σ2,
        )
        tensordivergence!(du_les, σ_all, false)
        project!(du_les, p_les, poisson_les)
        foreach(i -> axpy!(dt, du_les[i].data, u_c[i].data), 1:D)

        # Classic+Flux
        σ2 = stresstensor(ubar_coarse, visc)
        σ_cf = stresstensor(u_cf, visc)
        σ_all = map(
            (σ_cf, σ1, σ2) -> LazyField(
                (σ_cf, σ1, σ2, I) -> σ_cf[I] + σ1[I] - σ2[I],
                g_les,
                σ_cf,
                σ1,
                σ2,
            ),
            σ_cf,
            σΔH,
            σ2,
        )
        tensordivergence!(du_les, σ_all, false)
        project!(du_les, p_les, poisson_les)
        foreach(i -> axpy!(dt, du_les[i].data, u_cf[i].data), 1:D)

        # Classic+Flux+Div
        σ2 = stresstensor(ubar_coarse, visc)
        σ_cfd = stresstensor(u_cfd, visc)
        σ_both = map(
            (σ_cfd, σ2) ->
                LazyField((σ_cfd, σ2, I) -> σ_cfd[I] - σ2[I], g_les, σ_cfd, σ2),
            σ_cfd,
            σ2,
        )
        tensordivergence!(du_les, σ_both, false) # Overwrite du
        tensordivergence_nonsym!(du_les, rΔHi, true) # Add to existing du, don't project
        project!(du_les, p_les, poisson_les)
        foreach(i -> axpy!(dt, du_les[i].data, u_cfd[i].data), 1:D)

        # Classic+Flux+Div symmetrized
        σ2 = stresstensor(ubar_coarse, visc)
        σ_cfd_symm = stresstensor(u_cfd_symm, visc)
        σ_both = map(
            (σ1, σ2) -> LazyField((σ1, σ2, I) -> σ1[I] - σ2[I], g_les, σ1, σ2),
            σ_cfd_symm,
            σ2,
        )
        tensordivergence!(du_les, σ_both, false) # Overwrite du
        tensordivergence!(du_les, rΔHi_symm, true) # Add to existing du
        project!(du_les, p_les, poisson_les) # σ_both is projected
        foreach(i -> axpy!(dt, du_les[i].data, u_cfd_symm[i].data), 1:D)

        # DNS
        foreach(i -> axpy!(dt, du_dns[i].data, u_dns[i].data), 1:D)
    end

    # Filter
    for i = 1:D
        ubari = lazyfilter(u_dns[i], GΔH)
        coarsegrain!(ubar_coarse[i], ubari, faces[i])
    end
    project!(ubar_coarse, p_les, poisson_les)

    (; u_dns, u_les = ubar_coarse, u_nomo, u_c, u_cf, u_cfd, u_cfd_symm)
end

function compute_errors(uaid)
    g_dns = uaid.u_dns[1].grid
    g_les = uaid.u_nomo[1].grid
    D = dim(g_dns)
    e = map((; uaid.u_nomo, uaid.u_c, uaid.u_cf, uaid.u_cfd_symm, uaid.u_cfd)) do ubar
        sum(1:D) do i
            a = ubar[i].data
            b = uaid.u_les[i].data
            sum(abs2, a - b) / sum(abs2, b)
        end / D |> sqrt
    end
end

function plot_spectra(uaid)
    visc = 3e-4
    g_dns = uaid.u_dns[1].grid
    g_les = uaid.u_les[1].grid
    backend = defaultbackend()
    poisson_dns = poissonsolver(g_dns, backend)
    poisson_les = poissonsolver(g_les, backend)
    stuff_dns = spectral_stuff(g_dns, backend)
    stuff_les = spectral_stuff(g_les, backend)
    spec_dns = spectrum(uaid.u_dns, stuff_dns, poisson_dns)
    spec_les = map([
        (:u_les, "Filtered DNS"),
        (:u_nomo, "No-model"),
        (:u_c, "Classic"),
        (:u_cf, "Classic + Flux"),
        (:u_cfd, "Classic + Flux + Div"),
        (:u_cfd_symm, "Classic + Flux + Div (symm)"),
    ]) do (key, label)
        s = spectrum(uaid[key], stuff_les, poisson_les)
        (; s..., label)
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "k", ylabel = "E(k)", xscale = log10, yscale = log10)
    # lines!(ax, spec_dns.k, spec_dns.s; label = "DNS")
    foreach(s -> lines!(ax, s.k, s.s; s.label), spec_les)
    kkol = [10^1.5, maximum(spec_dns.k)]
    ekol = 1e4 * kkol .^ (-3)
    # lines!(ax, kkol, ekol)
    Legend(fig[1, 2], ax)
    save("~/toto.png" |> expanduser, fig)
    fig
end

end
