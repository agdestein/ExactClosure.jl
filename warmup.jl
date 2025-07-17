if false
    include("src/ExactClosure.jl")
    using .ExactClosure
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
using ExactClosure: NavierStokes
using Turbulox
using WGLMakie

@flushinfo "Loading case"

# Choose case
case = NavierStokes.smallcase() # Laptop CPU with 16 GB RAM
case = NavierStokes.mediumcase() # GPU with 24 GB RAM
case = NavierStokes.largecase() # GPU with 90 GB RAM (H100)

(; seed, viscosity, outdir, datadir, plotdir, totalenergy, kpeak) = case
grid = case.g_dns
T = typeof(grid.L)
case |> pairs

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

u = ustart

doplot = true
if doplot
    ut_obs = NavierStokes.plotsol(u, cache.p, viscosity)
end

@flushinfo "Running DNS warm-up"

# Burn-in
let
    t = 0.0 |> T
    cfl = 0.85 |> T
    tstop = 0.5 |> T
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

# @flushinfo "Plotting solution"
# CairoMakie.activate!()
# ut_obs = NavierStokes.plotsol(u, cache.p, viscosity)
# save("plotsol.pdf", current_figure())
