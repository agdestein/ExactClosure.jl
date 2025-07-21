# ExactClosure

Source code for the paper "Exact closure for discrete large-eddy simulation".

## Setup

Run

```
julia --project -e 'using Pkg; Pkg.develop(; url = "https://github.com/agdestein/Turbulox.jl")
```

to install all dependencies.

## Burgers equation

Run `burgers.jl`.
For multithreading, do
```
julia --project -t auto burgers.jl
```

## Navier-Stokes

Run `warmup.jl` to generate initial conditions.

Run `dns_aid.jl` for DNS-aided LES.

Run `process.jl` to postprocess the results.

Run `plots.jl` to generate plots.

In each script, choose the test case to run.
The paper uses `largecase`, but `smallcase` can be run on a laptop CPU.
Comment out the other cases.

```
case = NavierStokes.smallcase() # Laptop CPU with 16 GB RAM
case = NavierStokes.mediumcase() # GPU with 24 GB RAM
case = NavierStokes.largecase() # GPU with 90 GB RAM (H100)
```
