using Turbulox
using Random

comp = 4
nH = 16
nh = nH * comp
gh = Grid(; order = 2, dim = 2, n = nh, L = 1.0)
gH = Grid(; order = 2, dim = 2, n = nH, L = 1.0)

solver_h = poissonsolver(gh)
solver_H = poissonsolver(gH)

uh = vectorfield(gh);
uH = vectorfield(gH);
randn!(uh);

Turbulox.surfacefilter!(uH, uh, gH, comp)

ph = scalarfield(gh)
apply!(divergence!, gh, ph, uh)
ph

let
    u = randomfield(gh, solver_h)
    # u = vectorfield(gh)
    # randn!(u)
    πu = copy(u)
    fπu = vectorfield(gH)
    πfu = vectorfield(gH)
    ph = scalarfield(gh)
    pH = scalarfield(gH)
    Turbulox.surfacefilter!(πfu, u, gH, comp)
    project!(πfu, pH, solver_H, gH)
    project!(πu, ph, solver_h, gh)
    Turbulox.surfacefilter!(fπu, πu, gH, comp)
    πfu - fπu
end

let
    uh = vectorfield(gh)
    uH = vectorfield(gH)
    ph = scalarfield(gh)
    pH = scalarfield(gH)
    randn!(uh)
    project!(uh, ph, solver_h, gh)
    apply!(divergence!, gh, ph, uh)
    Turbulox.surfacefilter!(uH, uh, gH, comp)
    πuH = copy(uH)
    project!(uH, pH, solver_H, gH)
    πuH - uH
end
