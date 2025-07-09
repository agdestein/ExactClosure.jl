if false
    include("src/StructuralClosure.jl")
    using .StructuralClosure
end

# For logging to stderr without delay on SLURM
macro flushinfo(msg)
    esc(quote
        @info $msg
        flush(stderr)
    end)
end

@flushinfo "Loading packages"

using CairoMakie
using CUDA
using JLD2
using Random
using StructuralClosure: NavierStokes
using Turbulox
using WGLMakie

@flushinfo "Loading case"

case = NavierStokes.smallcase()
case = NavierStokes.largecase()
case = NavierStokes.newcase()
case = NavierStokes.snelliuscase()
(; seed, grid, viscosity, outdir, datadir, plotdir, totalenergy, kpeak) = case
T = typeof(grid.L)

ut_obs = u = ustart = cache = poisson = nothing
GC.gc();
CUDA.reclaim();
poisson = poissonsolver(grid);
ustart = Turbulox.randomfield_shell(
    Turbulox.energyprofile(kpeak),
    grid,
    poisson;
    rng = Xoshiro(seed),
    totalenergy,
);
cache = (; ustart = VectorField(grid), du = VectorField(grid), p = ScalarField(grid));

# Total initial energy
etot = sum(abs2, ustart.data) / 2 / grid.n^3
open(joinpath(outdir, "initial_energy.txt"), "w") do io
    println(io, etot)
end

u = ustart

doplot = true
if doplot
    ut_obs = NavierStokes.plotsol(u, cache.p, viscosity)
end

@flushinfo "Running burn-in"

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
        @flushinfo join(
            [
                "t = $(round(t; sigdigits = 4))",
                "Δt = $(round(Δt; sigdigits = 4))",
                "umax = $(round(maximum(abs, u.data); sigdigits = 4))",
            ],
            ",\t",
        )
        if doplot && i % 5 == 0
            setindex!(ut_obs, (; u, t))
            sleep(0.01)
        end
    end
    t
end

file = joinpath(outdir, "u.jld2")
@flushinfo "Saving final velocity field to $file"
save_object(file, u.data |> Array)
