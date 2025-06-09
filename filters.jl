using CairoMakie

tophat(x, Δ) = (abs(x) ≤ Δ / 2) / Δ
gaussian(x, Δ) = sqrt(6 / π) / Δ * exp(-6 * x^2 / Δ^2)
cutoff(x, Δ) = sinpi(2x / Δ) / π / x

f_tophat(k, Δ) = sinpi(k * Δ) / π / k / Δ
f_gaussian(k, Δ) = exp(-π^2 * k^2 * Δ^2 / 6)
f_cutoff(k, Δ) = k ≤ 1 / 2Δ

let
    Δ = 0.01
    w = 3
    n = 200
    x = range(-w * Δ, w * Δ, n)
    K = round(Int, 1 / Δ)
    k = 0:K
    fig = Figure(; size = (800, 350))
    ax = Axis(fig[1, 1]; title = "Convolution kernel", xlabel = "x / Δ", ylabel = "g * Δ")
    lines!(ax, x / Δ, tophat.(x, Δ) * Δ; label = "Top-hat")
    lines!(ax, x / Δ, gaussian.(x, Δ) * Δ; label = "Gaussian")
    lines!(ax, x / Δ, cutoff.(x, Δ) * Δ; label = "Cutoff")
    axislegend(ax)
    ax = Axis(fig[1, 2]; title = "Transfer function", xlabel = "k * Δ", ylabel = "ghat")
    scatterlines!(ax, k * Δ, f_tophat.(k, Δ); label = "Top-hat")
    scatterlines!(ax, k * Δ, f_gaussian.(k, Δ); label = "Gaussian")
    scatterlines!(ax, k * Δ, f_cutoff.(k, Δ); label = "Cutoff")
    axislegend(ax)
    save("filters.pdf", fig; backend = CairoMakie)
    fig
end
