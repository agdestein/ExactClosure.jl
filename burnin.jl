if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

using CairoMakie
using CUDA
using JLD2
using Random
using StructuralClosure: NavierStokes
using Turbulox
using WGLMakie

case = NavierStokes.smallcase()
# case = NavierStokes.largecase()
(; seed, grid, viscosity, outdir, datadir, plotdir, amplitude, kpeak) = case
T = typeof(grid.L)

poisson = poissonsolver(grid);
ustart = Turbulox.randomfield_simple(
    Turbulox.energyprofile,
    grid,
    poisson;
    rng = Xoshiro(seed),
    params = (; kpeak, amplitude),
);
cache = (; ustart = VectorField(grid), du = VectorField(grid), p = ScalarField(grid));

# Total initial energy
etot = sum(abs2, ustart.data) / 2 / grid.n^3
open(joinpath(outdir, "initial_energy.txt"), "w") do io
    println(io, etot)
end

u = ustart
# u = VectorField(grid, copy(ustart.data));

doplot = false
if doplot
    ut_obs = NavierStokes.plotsol(u, cache.p, viscosity)
end

# Burn-in
let
    t = 0.0 |> T
    cfl = 0.85 |> T
    tstop = 0.4 |> T
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * propose_timestep(u, viscosity)
        Δt = min(Δt, tstop - t)
        timestep!(right_hand_side!, u, cache, Δt, poisson; viscosity)
        t += Δt
        @info "t = $t,\tumax = $(maximum(abs, u.data))"
        if doplot && i % 5 == 0
            setindex!(ut_obs, (; u, t))
            sleep(0.01)
        end
    end
    t
end

save_object(joinpath(outdir, "u.jld2"), u.data |> Array)
