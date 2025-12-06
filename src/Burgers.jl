"1D simulation."
module Burgers

export Grid,
    points_stag,
    points_coll,
    δ_stag,
    δ_coll,
    stress,
    cfl,
    timestep!,
    volavg_stag!,
    volavg_coll!,
    suravg_stag!,
    suravg_coll!,
    randomfield,
    dissipation,
    dissipation!,
    spacing

using FFTW
using LinearAlgebra
using Random
import AcceleratedKernels as AK

"Uniform grid of length `L` and `n` volumes."
struct Grid{T}
    L::T
    n::Int
end

"Boundary conditions: `g(i)` gives periodic index."
@inline (g::Grid)(i) = mod1(i, g.n)

"Grid spacing."
@inline spacing(g::Grid) = g.L / g.n

# Grid points
points_stag(g) = range(0, g.L, g.n + 1)[2:end]
points_coll(g) = range(0, g.L, g.n + 1)[2:end] .- spacing(g) / 2

# Finite difference
@inline δ_stag(g, u, i) = (u[i|>g] - u[i-1|>g]) / spacing(g)
@inline δ_coll(g, p, i) = (p[i+1|>g] - p[i|>g]) / spacing(g)

@inline stress(g, u, visc, i) = -(u[i-1|>g] + u[i])^2 / 8 + visc * δ_stag(g, u, i)

# CFL number
cfl(g, u, visc) = min(spacing(g) / maximum(abs, u), spacing(g)^2 / visc)

"Perform one time step (inplace)."
function timestep!(g, u, s, visc, dt)
    AK.foreachindex(u) do i
        s[i] = stress(g, u, visc, i)
    end
    AK.synchronize(AK.get_backend(u))
    AK.foreachindex(u) do i
        u[i] += dt * δ_coll(g, s, i)
    end
    AK.synchronize(AK.get_backend(u))
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
export gaussian_weights

function tophat_weights(g, comp)
    R = div(comp, 2)
    w = fill(one(g.L) / comp, comp)
    R, w
end
export tophat_weights

function convolution!(g, kernel, ubar, u)
    R, w = kernel
    AK.foreachindex(ubar) do i
        s = zero(eltype(u))
        for r = (-R):R
            @inbounds s += w[r+R+1] * u[i+r|>g]
        end
        @inbounds ubar[i] = s
    end
    AK.synchronize(AK.get_backend(ubar))
    ubar
end
export convolution!

function coarsegrain_convolve_stag!(gH, gh, uH, uh, kernel)
    comp = div(gh.n, gH.n)
    R, w = kernel
    AK.foreachindex(uH) do i
        s = zero(eltype(uH))
        for r = (-R):R
            @inbounds s += w[R+1+r] * uh[i*comp+r|>gh]
        end
        @inbounds uH[i] = s
    end
    AK.synchronize(AK.get_backend(uH))
    uH
end
export coarsegrain_convolve_stag!

function coarsegrain_convolve_coll!(gH, gh, pH, ph, kernel)
    comp = div(gh.n, gH.n)
    a = div(comp, 2)
    R, w = kernel
    AK.foreachindex(pH) do i
        s = zero(eltype(pH))
        for r = (-R):R
            @inbounds s += w[R+1+r] * ph[i*comp-a+r|>gh]
        end
        @inbounds pH[i] = s
    end
    AK.synchronize(AK.get_backend(pH))
    pH
end
export coarsegrain_convolve_coll!

function volavg_stag!(gH, gh, uH, uh)
    comp = div(gh.n, gH.n)
    R = div(comp, 2)
    @assert 2 * R + 1 == comp "Use odd compression."
    AK.foreachindex(uH) do i
        s = zero(eltype(uH))
        for r = (-R):R
            @inbounds s += uh[i*comp+r|>gh] / comp
        end
        @inbounds uH[i] = s
    end
    AK.synchronize(AK.get_backend(uH))
    uH
end

function volavg_coll!(gH, gh, pH, ph)
    comp = div(gh.n, gH.n)
    AK.foreachindex(pH) do i
        s = zero(eltype(pH))
        for j = 1:comp
            @inbounds s += ph[(i-1)*comp+j] / comp
        end
        @inbounds pH[i] = s
    end
    AK.synchronize(AK.get_backend(pH))
    pH
end

function suravg_stag!(gH, gh, uH, uh)
    comp = div(gh.n, gH.n)
    AK.foreachindex(uH) do i
        @inbounds uH[i] = uh[i*comp]
    end
    AK.synchronize(AK.get_backend(uH))
    uH
end

function suravg_coll!(gH, gh, pH, ph)
    comp = div(gh.n, gH.n)
    R = div(comp, 2)
    @assert 2 * R + 1 == comp "Use odd compression."
    AK.foreachindex(pH) do i
        @inbounds pH[i] = ph[i*comp-R]
    end
    AK.synchronize(AK.get_backend(pH))
    pH
end

function randomfield(rng, g, kpeak, amp)
    k = 0:div(g.n, 2)
    c = @. amp * (k / kpeak)^2 * exp(-(k / kpeak)^2 / 2 + 2π * im * rand(rng))
    irfft(c * g.n, g.n)
end

dissipation(g, u, s) =
    map(1:g.n) do i
        δ_stag(g, u, i) * s[i]
        # u[i] * δ_coll(g, s, i)
    end

function dissipation!(g, d, u, s)
    AK.foreachindex(d) do i
        @inbounds d[i] = s[i] * δ_stag(g, u, i)
    end
    AK.synchronize(AK.get_backend(d))
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

# Compare closure formulations
function dns_aided_les(ustart, gh, gH, visc; Δ, tstop, cfl_factor, lesfiltertype)
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

    # Initial double-filtered state
    uH = zeros(gH.n)
    coarsegrain_convolve_stag!(gH, gh, uH, uh, doublekernel)

    # Allocate arrays
    u = (;
        dns_ref = copy(uh),
        dns_fil = copy(uH),
        nomodel = copy(uH),
        class_m = copy(uH), # bar(r(u)) - r(bar(u))
        class_p = copy(uH), # bar(r(u)) - rh(bar(u))
        swapfil = copy(uH),
    )
    su = zeros(gh.n)
    fsu = zeros(gH.n)
    vsu = zeros(gH.n)
    vu = zeros(gH.n)
    VU = zeros(gh.n) # Same as vu, but on DNS grid
    σ_nomodel = zeros(gH.n)
    σ_class_m = zeros(gH.n)
    σ_class_p = zeros(gH.n)
    σ_swapfil = zeros(gH.n)

    # Time stepping
    t = 0.0
    while t < tstop
        # Get time step
        dt = cfl_factor * cfl(gh, u.dns_ref, visc)
        dt = min(dt, tstop - t) # Don't overstep

        # DNS stress
        AK.foreachindex(su) do i
            @inbounds su[i] = stress(gh, u.dns_ref, visc, i)
        end
        AK.synchronize(AK.get_backend(su))

        # Filtered stresses
        coarsegrain_convolve_coll!(gH, gh, fsu, su, leskernel)
        coarsegrain_convolve_coll!(gH, gh, vsu, su, doublekernel)

        # Filtered velocity
        coarsegrain_convolve_stag!(gH, gh, vu, u.dns_ref, doublekernel)
        convolution!(gh, doublekernel, VU, u.dns_ref)

        # LES stresses
        comp = div(gh.n, gH.n)
        a = div(comp, 2)
        AK.foreachindex(uH) do i
            @inbounds svu = stress(gH, vu, visc, i)
            @inbounds sVU = stress(gh, VU, visc, i * comp - a)
            @inbounds σ_nomodel[i] = stress(gH, u.nomodel, visc, i)
            @inbounds σ_class_m[i] = stress(gH, u.class_m, visc, i) + vsu[i] - sVU
            @inbounds σ_class_p[i] = stress(gH, u.class_p, visc, i) + vsu[i] - svu
            @inbounds σ_swapfil[i] = stress(gH, u.swapfil, visc, i) + fsu[i] - svu
        end
        AK.synchronize(AK.get_backend(uH))

        # DNS time step
        AK.foreachindex(uh) do i
            @inbounds u.dns_ref[i] += dt * δ_coll(gh, su, i)
        end
        AK.synchronize(AK.get_backend(uh))

        # LES time steps
        AK.foreachindex(uH) do i
            @inbounds u.nomodel[i] += dt * δ_coll(gH, σ_nomodel, i)
            @inbounds u.class_m[i] += dt * δ_coll(gH, σ_class_m, i)
            @inbounds u.class_p[i] += dt * δ_coll(gH, σ_class_p, i)
            @inbounds u.swapfil[i] += dt * δ_coll(gH, σ_swapfil, i)
        end
        AK.synchronize(AK.get_backend(uh))

        # Time step
        t += dt
    end

    # Final filtered velocity
    coarsegrain_convolve_stag!(gH, gh, u.dns_fil, u.dns_ref, doublekernel)

    u
end

function compute_dissipation(series, setup, lesfiltertype)
    (; visc, Δ_ratio) = setup
    map(series) do (; nH, fields)
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
            AK.foreachindex(su) do i
                @inbounds su[i] = stress(gh, uh, visc, i)
            end
            AK.synchronize(AK.get_backend(su))

            # Filtered stresses
            coarsegrain_convolve_coll!(gH, gh, fsu, su, leskernel)
            coarsegrain_convolve_coll!(gH, gh, vsu, su, doublekernel)

            # Filtered velocity
            coarsegrain_convolve_stag!(gH, gh, vu, uh, doublekernel)
            convolution!(gh, doublekernel, VU, uh)

            # LES stresses
            comp = div(gh.n, gH.n)
            a = div(comp, 2)
            AK.foreachindex(vu) do i
                @inbounds svu = stress(gH, vu, visc, i)
                @inbounds sVU = stress(gh, VU, visc, i * comp - a)
                @inbounds σ_class_m[i] = vsu[i] - sVU
                @inbounds σ_class_p[i] = vsu[i] - svu
                @inbounds σ_swapfil[i] = fsu[i] - svu
            end
            AK.synchronize(AK.get_backend(vu))

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
export compute_dissipation

function create_dns(setup; cfl_factor)
    (; L, nh, kp, a, visc, tstop, nsample) = setup
    g = Grid(L, nh)
    Ustart = zeros(nh, nsample)
    U = zeros(nh, nsample)
    for i = 1:nsample
        @info "DNS sample $i of $nsample"
        ustart = randomfield(Xoshiro(i), g, kp, a)
        u = copy(ustart)
        s = zero(ustart)
        t = 0.0
        while t < tstop
            dt = cfl_factor * cfl(g, u, visc) # Propose timestep
            dt = min(dt, tstop - t) # Don't overstep
            timestep!(g, u, s, visc, dt) # Perform timestep
            t += dt
        end
        Ustart[:, i] = ustart
        U[:, i] = u
    end
    Ustart, U
end
export create_dns

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
            AK.foreachindex(uh) do i
                σh[i] = stress(gh, uh, visc, i)
            end
            convolution!(gh, doublekernel, σh_double1, σh)
            convolution!(gh, doublekernel, uh_double, uh)
            AK.foreachindex(uh) do i
                σh_double2[i] = stress(gh, uh_double, visc, i)
                th[i] = σh_double1[i] - σh_double2[i]
            end

            # Compute Smagorinsky stress
            AK.foreachindex(uh_double) do i
                δu = δ_stag(gh, uh_double, i)
                sh[i] = -(Δ^2 + H^2) * abs(δu) * δu
            end

            # Compute dissipation coefficients
            AK.foreachindex(uh_double) do i
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
            AK.foreachindex(uh) do i
                σh[i] = stress(gh, uh, visc, i)
            end
            convolution!(gh, leskernel, σh_double1, σh)
            convolution!(gh, doublekernel, uh_double, uh)
            AK.foreachindex(uh) do i
                # Coarse-grid stress (on the fine grid)
                uleft = uh_double[i-a-1|>gh]
                uright = uh_double[i+a|>gh]
                σh_double2[i] =
                    -(uleft + uright)^2 / 8 + visc * (uright - uleft) / spacing(gH)
                tH[i] = σh_double1[i] - σh_double2[i]
            end

            # Compute Smagorinsky stress
            AK.foreachindex(uh_double) do i
                # Coarse-grid derivative (on fine grid)
                uleft = uh_double[i-a-1|>gh]
                uright = uh_double[i+a|>gh]
                δu = (uright - uleft) / spacing(gH)
                sH[i] = -(Δ^2 + H^2) * abs(δu) * δu
            end

            # Compute dissipation coefficients
            AK.foreachindex(uh_double) do i
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
                # AK.foreachindex(ubar) do i
                #     ubar[i] += dt * k1[i]
                # end

                # RK4 step
                copyto!(ubar0, ubar)
                smagorinsky_rhs!(k1, ubar, s, gH, visc, C)
                AK.foreachindex(ubar) do i
                    ubar[i] = ubar0[i] + dt / 2 * k1[i]
                end
                smagorinsky_rhs!(k2, ubar, s, gH, visc, C)
                AK.foreachindex(ubar) do i
                    ubar[i] = ubar0[i] + dt / 2 * k2[i]
                end
                smagorinsky_rhs!(k3, ubar, s, gH, visc, C)
                AK.foreachindex(ubar) do i
                    ubar[i] = ubar0[i] + dt * k3[i]
                end
                smagorinsky_rhs!(k4, ubar, s, gH, visc, C)
                AK.foreachindex(ubar) do i
                    ubar[i] = ubar0[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                end
            end
            Ubar[:, j] = ubar
        end

        Ubar
    end
end

function smagorinsky_rhs!(du, u, s, g, visc, C)
    AK.foreachindex(u) do i
        δu = δ_stag(g, u, i)
        smag = -C * abs(δu) * δu
        s[i] = stress(g, u, visc, i) - smag
    end
    AK.foreachindex(u) do i
        du[i] = δ_coll(g, s, i)
    end
end

end
