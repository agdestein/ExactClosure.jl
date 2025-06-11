module StructuralClosure

using Makie

include("Burgers.jl")
include("NavierStokes.jl")

export Burgers, NavierStokes

"Plot zoom-in box."
function zoombox!(subfig, ax1; point, logx, logy, relwidth, relheight)
    # sk, se = sqrt(logx), sqrt(logy)
    sk, se = logx, logy
    kk, ee = point
    k0, k1 = kk / sk, kk * sk
    e0, e1 = ee / se, ee * se
    limits = (k0, k1, e0, e1)
    lines!(
        ax1,
        [
            Point2f(k0, e0),
            Point2f(k1, e0),
            Point2f(k1, e1),
            Point2f(k0, e1),
            Point2f(k0, e0),
        ];
        color = :black,
        linewidth = 1.5,
    )
    ax2 = Axis(
        subfig;
        width = Relative(relwidth),
        height = Relative(relheight),
        halign = 0.05,
        valign = 0.10,
        limits,
        xscale = log10,
        yscale = log10,
        xticksvisible = false,
        xticklabelsvisible = false,
        xgridvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        ygridvisible = false,
        backgroundcolor = :white,
    )
    # https://discourse.julialang.org/t/makie-inset-axes-and-their-drawing-order/60987/5
    translate!(ax2.scene, 0, 0, 10)
    translate!(ax2.elements[:background], 0, 0, 9)
    ax2
end


end
