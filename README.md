# ExactClosure

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://agdestein.github.io/ExactClosure.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://agdestein.github.io/ExactClosure.jl/dev/)
[![Build Status](https://github.com/agdestein/ExactClosure.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/agdestein/ExactClosure.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/agdestein/ExactClosure.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/agdestein/ExactClosure.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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

First generate initial conditions by running `burnin.jl`.

Then run DNS-aided LES with `dns_aid.jl`.
