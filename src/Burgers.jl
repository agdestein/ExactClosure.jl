"1D simulation."
module Burgers

using FFTW

struct Grid{T}
    L::T
    n::Int
end

# Boundary conditions: g(i) gives periodic index
@inline (g::Grid)(i) = mod1(i, g.n)

# Grid size
h(g) = g.L / g.n

# Grid points
points_stag(g) = range(0, g.L, g.n + 1)[2:end]
points_coll(g) = range(0, g.L, g.n + 1)[2:end] .- h(g) / 2

# Finite difference
@inline δ_stag(g, u, i) = (u[i|>g] - u[i-1|>g]) / h(g)
@inline δ_coll(g, p, i) = (p[i+1|>g] - p[i|>g]) / h(g)

@inline stress(g, u, visc, i) = -(u[i-1|>g] + u[i])^2 / 8 + visc * δ_stag(g, u, i)
@inline convstress(g, u, visc, i) = -(u[i-1|>g] + u[i])^2 / 8

# CFL number
cfl(g, u, visc) = min(h(g) / maximum(abs.(u)), h(g)^2 / visc)

# Perform one time step
function timestep(g, u, visc, dt)
    s = map(i -> stress(g, u, visc, i), 1:g.n)
    map(i -> u[i] + dt * δ_coll(g, s, i), 1:g.n)
end

# Filters

function volavg_stag!(gH, gh, uH, uh)
    comp = div(gh.n, gH.n)
    R = div(comp, 2)
    @assert 2 * R + 1 == comp "Use odd compression."
    for i = 1:gH.n
        s = zero(eltype(uH))
        for r = (-R):R
            s += uh[i*comp+r|>gh] / comp
        end
        uH[i] = s
    end
    uH
end

function volavg_coll!(gH, gh, pH, ph)
    comp = div(gh.n, gH.n)
    for i = 1:gH.n
        s = zero(eltype(pH))
        for j = 1:comp
            s += ph[(i-1)*comp+j] / comp
        end
        pH[i] = s
    end
    pH
end

function suravg_stag!(gH, gh, uH, uh)
    comp = div(gh.n, gH.n)
    for i = 1:gH.n
        uH[i] = uh[i*comp]
    end
    uH
end

function suravg_coll!(gH, gh, pH, ph)
    comp = div(gh.n, gH.n)
    R = div(comp, 2)
    @assert 2 * R + 1 == comp "Use odd compression."
    for i = 1:gH.n
        pH[i] = ph[i*comp-R]
    end
    pH
end

function randomfield(g, kpeak, amp)
    k = 0:div(g.n, 2)
    c = @. amp * (k / kpeak)^2 * exp(-(k / kpeak)^2 / 2 + 2π * im * rand())
    # c .|> abs |> display
    # c = @. c + (abs(c) < 1e-16) * 1e-16
    # Main.Makie.scatter(k[2:end], abs2.(c)[2:end]; axis = (; xscale = log10, yscale = log10)) |> display
    irfft(c * g.n, g.n)
end

specavg3(e) =
    map(eachindex(e)) do i
        i in eachindex(e)[2:(end-1)] || return e[i]
        (e[i-1] + e[i] + e[i+1]) / 3
    end

specavg5(e) =
    map(eachindex(e)) do i
        i in eachindex(e)[3:(end-2)] || return e[i]
        (e[i-2] + e[i-1] + e[i] + e[i+1] + e[i+2]) / 5
    end

dissipation(g, u, s) =
    map(1:g.n) do i
        δ_stag(g, u, i) * s[i]
        # u[i] * δ_coll(g, s, i)
    end

# Compare closure formulations
function dns_aided_les(ustart, gh, gH, visc; tstop, cfl_factor)
    uh = ustart
    uH = zeros(gH.n)
    volavg_stag!(gH, gh, uH, uh)
    u = (;
        dns_ref = copy(uh),
        dns_fil = copy(uH),
        nomodel = copy(uH),
        classic = copy(uH),
        revolut = copy(uH),
        classic_conv = copy(uH),
        revolut_conv = copy(uH),
    )
    su = zeros(gh.n)
    cu = zeros(gh.n)
    fsu = zeros(gH.n)
    vsu = zeros(gH.n)
    fcu = zeros(gH.n)
    vcu = zeros(gH.n)
    vu = zeros(gH.n)
    svu = zeros(gH.n)
    cvu = zeros(gH.n)
    σ_nomodel = zeros(gH.n)
    σ_classic = zeros(gH.n)
    σ_revolut = zeros(gH.n)
    σ_classic_conv = zeros(gH.n)
    σ_revolut_conv = zeros(gH.n)
    t = 0.0
    while t < tstop
        dt = cfl_factor * cfl(gh, u.dns_ref, visc)
        dt = min(dt, tstop - t) # Don't overstep
        for i = 1:gh.n
            su[i] = stress(gh, u.dns_ref, visc, i)
            cu[i] = convstress(gh, u.dns_ref, visc, i)
        end
        suravg_coll!(gH, gh, fsu, su)
        volavg_coll!(gH, gh, vsu, su)
        suravg_coll!(gH, gh, fcu, cu)
        volavg_coll!(gH, gh, vcu, cu)
        volavg_stag!(gH, gh, vu, u.dns_ref)
        for i = 1:gH.n
            svu = stress(gH, vu, visc, i)
            cvu = convstress(gH, vu, visc, i)
            σ_nomodel[i] = stress(gH, u.nomodel, visc, i)
            σ_classic[i] = stress(gH, u.classic, visc, i) + vsu[i] - svu
            σ_revolut[i] = stress(gH, u.revolut, visc, i) + fsu[i] - svu
            σ_classic_conv[i] = stress(gH, u.classic_conv, visc, i) + vcu[i] - cvu
            σ_revolut_conv[i] = stress(gH, u.revolut_conv, visc, i) + fcu[i] - cvu
        end
        for i = 1:gh.n
            u.dns_ref[i] += dt * δ_coll(gh, su, i)
        end
        for i = 1:gH.n
            u.nomodel[i] += dt * δ_coll(gH, σ_nomodel, i)
            u.classic[i] += dt * δ_coll(gH, σ_classic, i)
            u.revolut[i] += dt * δ_coll(gH, σ_revolut, i)
            u.classic_conv[i] += dt * δ_coll(gH, σ_classic_conv, i)
            u.revolut_conv[i] += dt * δ_coll(gH, σ_revolut_conv, i)
        end
        t += dt
    end
    volavg_stag!(gH, gh, u.dns_fil, u.dns_ref)
    u
end

export Grid,
    points_stag,
    points_coll,
    δ_stag,
    δ_coll,
    convstress,
    stress,
    cfl,
    timestep,
    volavg_stag!,
    volavg_coll!,
    suravg_stag!,
    suravg_coll!,
    randomfield,
    specavg3,
    specavg5,
    dissipation

end
