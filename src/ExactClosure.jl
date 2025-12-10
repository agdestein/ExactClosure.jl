module ExactClosure

using Makie

include("zoombox.jl")
include("Burgers.jl")
include("NavierStokes.jl")

export Burgers, NavierStokes

end
