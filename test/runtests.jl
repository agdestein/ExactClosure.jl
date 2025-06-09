using StructuralClosure
using Test
using Aqua
using JET

@testset "StructuralClosure.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(StructuralClosure; project_extras = false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(StructuralClosure; target_defined_modules = true)
    end
    # Write your tests here.
end
