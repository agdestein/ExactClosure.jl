# This is just a hack for "go to definition" to work in editor.
if false
    include("src/ExactClosure.jl")
    using .ExactClosure
end

@info "Loading packages"
flush(stderr)

using Adapt
using CairoMakie
using CUDA
using JLD2
using KernelDensity
using LinearAlgebra
using Random
using ExactClosure: NavierStokes
using Turbulox
using Turbulox.KernelAbstractions
using WGLMakie
using WriteVTK

@info "Loading case"
flush(stderr)

# Choose case
case = NavierStokes.smallcase() # Laptop CPU with 16 GB RAM
case = NavierStokes.mediumcase() # GPU with 24 GB RAM
case = NavierStokes.largecase() # GPU with 90 GB RAM (H100)

(; viscosity, outdir, datadir, plotdir, seed, g_dns, g_les) = case
T = typeof(g_dns.L)

poisson_dns = poissonsolver(g_dns);

ustart = let
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    VectorField(g_dns, data)
end

s = get_scale_numbers(ustart, viscosity)
s |> pairs
1 / s.λ

false && let
    @info "Computing Q-criterion"
    flush(stderr)
    u = ustart
    (; grid) = u
    compression = 5
    n = div(compression, 2) + 1
    fu = VectorField(grid)
    unorm = ScalarField(grid)
    funorm = ScalarField(grid)
    qu = ScalarField(grid)
    qfu = ScalarField(grid)
    qfu_coarse = ScalarField(grid)
    Turbulox.volumefilter_explicit!(fu, u, compression)
    apply!(Turbulox.velocitynorm!, grid, unorm, u)
    apply!(Turbulox.velocitynorm!, grid, funorm, fu)
    ∇u = LazyScalarField(grid, Turbulox.∇_collocated, 1, u)
    ∇fu = LazyScalarField(grid, Turbulox.∇_collocated, 1, fu)
    ∇fu_coarse = LazyScalarField(grid, Turbulox.∇_collocated, n, fu)
    apply!(Turbulox.compute_q!, grid, qu, ∇u)
    apply!(Turbulox.compute_q!, grid, qfu, ∇fu)
    apply!(Turbulox.compute_q!, grid, qfu_coarse, ∇fu_coarse)
    x = get_axis(u.grid, Coll())
    file = joinpath(outdir, "data", "q")
    @info "Writing Q to $file"
    flush(stderr)
    vtk_grid(file, x, x, x) do vtk
        vtk["Qh(u)"] = qu.data |> Array
        vtk["Qh(ubar)"] = qfu.data |> Array
        vtk["QH(ubar)"] = qfu_coarse.data |> Array
        vtk["norm(u)"] = unorm.data |> Array
        vtk["norm(ubar)"] = funorm.data |> Array
    end
end

let
    @info "Computing DNS spectrum after warm-up"
    flush(stderr)
    # Load DNS snapshot
    path = joinpath(outdir, "u.jld2")
    data = path |> load_object |> adapt(g_dns.backend)
    u = VectorField(g_dns, data)
    # Spectra
    dns = spectrum(u)
    les = map(g_les) do g_les
        compression = div(g_dns.n, g_les.n)
        v = VectorField(g_les)
        Turbulox.volumefilter!(v, u, compression)
        spectrum(v)
    end
    # Dissipation coefficient
    diss = ScalarField(g_dns)
    apply!(Turbulox.dissipation!, g_dns, diss, u, viscosity)
    D = sum(diss.data) / length(diss)
    diss = nothing # free up memory
    # Save
    file = joinpath(datadir, "warm-up-spectra.jld2")
    @info "Saving warm-up spectrum to $file"
    flush(stderr)
    jldsave(file; dns, les, D)
end


let
    @info "Computing dissipation coefficient density"
    flush(stderr)
    u = ustart
    for g_les in g_les
        compression = div(g_dns.n, g_les.n)
        poisson_les = poissonsolver(g_les)
        fu = VectorField(g_les)
        for experiment in ["volavg", "project_volavg", "surfavg"]
            @info "Experiment: $experiment"
            flush(stderr)
            if experiment == "volavg"
                sfs = NavierStokes.sfs_tensors_volume(;
                    ustart,
                    g_dns,
                    g_les,
                    poisson_dns,
                    poisson_les,
                    viscosity,
                    compression,
                    doproject = false,
                )
                volumefilter!(fu, u, compression)
            elseif experiment == "project_volavg"
                sfs = NavierStokes.sfs_tensors_volume(;
                    ustart,
                    g_dns,
                    g_les,
                    poisson_dns,
                    poisson_les,
                    viscosity,
                    compression,
                    doproject = true,
                )
                volumefilter!(fu, u, compression)
                p = ScalarField(g_les)
                project!(fu, p, poisson_les)
            elseif experiment == "surfavg"
                sfs = NavierStokes.sfs_tensors_surface(;
                    ustart,
                    g_dns,
                    g_les,
                    poisson_dns,
                    poisson_les,
                    viscosity,
                    compression,
                    doproject = false,
                )
                surfacefilter!(fu, u, compression)
            else
                error("Unknown experiment: $experiment")
            end
            diss = (;
                classic = ScalarField(g_les),
                swapfil = ScalarField(g_les),
                swapfil_symm = ScalarField(g_les),
            )
            apply!(
                Turbulox.tensordissipation_staggered!,
                g_les,
                diss.classic,
                sfs.σ_classic,
                fu,
            )
            apply!(
                Turbulox.tensordissipation_staggered!,
                g_les,
                diss.swapfil,
                sfs.σ_swapfil,
                fu,
            )
            apply!(
                Turbulox.tensordissipation_staggered!,
                g_les,
                diss.swapfil_symm,
                sfs.σ_swapfil_symm,
                fu,
            )
            diss = adapt(Array, diss)
            density = map(d -> (d.data |> vec |> kde |> x -> (; x.x, x.density)), diss)
            file = joinpath(datadir, "dissipation-$(experiment)-$(g_les.n).jld2")
            @info "Saving dissipation density to $file"
            flush(stderr)
            jldsave(file; density)
        end
    end
end
