# ExactClosure

Source code for the paper "Exact closure for discrete LES".

## Setup

```
julia --project -e 'using Pkg; Pkg.develop(; url = "https://github.com/agdestein/Turbulox.jl")
```

## Burgers equation

Run `burgers.jl`.
For multithreading, do
```
julia --project -t auto burgers.jl
```

## Navier-Stokes

Generate initial conditions by running `warmup.jl`.

Run DNS-aided LES with `dns_aid.jl`.

Run `plots.jl` to generate plots.
