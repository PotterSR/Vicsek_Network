using CairoMakie, DelimitedFiles

include("./Utils.jl")
using .Utils


function run_evolution_Vicsek!(n_steps, x, y, angle, parameters; save_every = 0, 
    save_parameter = false, save_positions = false)

    (; dt, v_0, Lx, Ly, r_0, η ,skin) = parameters

    n = length(x)

    r_list = r_0 + skin
    Mx, My = cell_grid(Lx, Ly, r_list)

    head      = zeros(Int, Mx*My)
    next      = zeros(Int, n)
    neighbors = [Int[] for _ in 1:n]

    build_verlet_list!(neighbors, head, next, x, y, Lx, Ly, Mx, My, r_list)
    x0 = copy(x)
    y0 = copy(y)
    n_rebuilds = 1

    order_parameter = Float64[]

    f_angle = zeros(n)
    z       = zeros(ComplexF64, n)
    acc     = zeros(ComplexF64, n)

    frames_x = [copy(x)]
    frames_y = [copy(y)]
    frames_angle = [copy(angle)]

    for step in 1:n_steps

        if needs_rebuild(x, y, x0, y0, skin, Lx, Ly)
            build_verlet_list!(neighbors, head, next, x, y, Lx, Ly, Mx, My, r_list)
            x0 .= x
            y0 .= y
            n_rebuilds += 1
        end 

        compute_angle_avg!(f_angle, z, acc, neighbors, x, y, angle, Lx, Ly; r_0 = r_0)

        for i in 1:n
            x[i] += (v_0*cos(angle[i]))*dt 
            y[i] += (v_0*sin(angle[i]))*dt


            angle[i] = f_angle[i] + (rand() - 0.5)*η
            angle[i] = mod(angle[i], 2*π)
        end

        if step%5000 == 0
            porcentaje = step/n_steps * 100
            println("η = $η: running $porcentaje %")
        end

        rescale_periodic!(x, Lx)
        rescale_periodic!(y, Ly)

        if save_every > 0 && step % save_every == 0 && save_positions
            push!(frames_x, copy(x))
            push!(frames_y, copy(y))
            push!(frames_angle, copy(angle))
        end


        if (save_every > 0 && step % save_every == 0 && save_parameter)
            push!(order_parameter, compute_order_parameter(angle))
        end

    end

    if save_parameter
        return n_rebuilds, frames_x, frames_y, frames_angle, order_parameter
    end
    
    return n_rebuilds, frames_x, frames_y, frames_angle
    
end


function make_vicsek_gif(frames_x, frames_y, frames_angle, Lx, Ly, dt, save_every;
                         filename = "vicsek.mp4", framerate = 20,
                         arrow_len = 0.6)

    pos(k) = Point2f.(frames_x[k], frames_y[k])
    dir(k) = Vec2f.(cos.(frames_angle[k]), sin.(frames_angle[k]))

    pos_obs = Observable(pos(1))
    dir_obs = Observable(dir(1))
    title   = Observable("t = 0.0")

    fig = Figure(size = (600, 600))
    ax  = Axis(fig[1, 1]; aspect = DataAspect(), title = title,
               limits = ((0, Lx), (0, Ly)))

    arrows2d!(ax, pos_obs, dir_obs;
              lengthscale = arrow_len, color = :black,
              shaftwidth = 1.0, tipwidth = 4.0, tiplength = 4.0)

    record(fig, filename, eachindex(frames_x); framerate = framerate) do k
        pos_obs[] = pos(k)
        dir_obs[] = dir(k)
        title[]   = "t = $(round((k - 1) * save_every * dt, digits = 2))"
    end

    return filename
end


function unwrap(x, L)
    u = copy(x)
    for k in 2:length(x)
        d = x[k] - x[k-1]
        u[k] = u[k-1] + d - L * round(d / L)   # minimum-image displacement
    end
    return u
end

# one full run at noise η: returns φ(t) and writes φ(t) and the final adjacency matrix to outdir
function simulate(η, settings, outdir)
    (; n, rho, dt, v_0, r_0, skin, n_steps, save_every) = settings

    Lx = Ly = sqrt(n/rho)

    x, y = positions_random(n; Lx = Lx, Ly = Ly)
    θ    = 2π .* rand(n)

    parameters = (dt=dt, v_0=v_0, Lx=Lx, Ly=Ly, r_0=r_0, η=η, skin=skin)

    n_rebuilds, _, _, _, P =
        run_evolution_Vicsek!(n_steps, x, y, θ, parameters; save_parameter = true, save_every = save_every)
    println("η = $η: Verlet list rebuilt $n_rebuilds times in $n_steps steps")

    t = (1:length(P)) .* save_every .* dt       # P is sampled every save_every steps
    writedlm(joinpath(outdir, "phi_eta_$η.csv"), hcat(t, P), ',')

    # x, y were updated in place, so they hold the final positions
    r_list = r_0 + skin
    Mx, My = cell_grid(Lx, Ly, r_list)
    head      = zeros(Int, Mx*My)
    next      = zeros(Int, n)
    neighbors = [Int[] for _ in 1:n]
    build_verlet_list!(neighbors, head, next, x, y, Lx, Ly, Mx, My, r_list)

    A = compute_adjacency_matrix(x, y, neighbors, Lx, Ly; r_0 = r_0)

    open(joinpath(outdir, "adjacency_eta_$η.txt"), "w") do f
        for i in 1:n
            for j in 1:n
                print(f, A[i, j], " ")
            end
            print(f, "\n")
        end
    end

    return P
end


settings = (n = 4000, rho = 2.0, dt = 0.1, v_0 = 0.03, r_0 = 1.0, skin = 0.3,
            n_steps = 10000, save_every = 10)

# noise values come from the command line: julia --project=. run.jl 0.5 1.0 2.0
η_values = isempty(ARGS) ? [0.5] : parse.(Float64, ARGS)

outdir = "results"
mkpath(outdir)

println("running $(length(η_values)) values of η on $(Threads.nthreads()) threads")

P_all = Vector{Vector{Float64}}(undef, length(η_values))

Threads.@threads for k in eachindex(η_values)          # each η is an independent run
    P_all[k] = simulate(η_values[k], settings, outdir)
end

# plotting is done serially: CairoMakie is not thread safe
for (η, P) in zip(η_values, P_all)
    t = (1:length(P)) .* settings.save_every .* settings.dt

    fig = Figure(size = (700, 450), fontsize = 16)
    ax  = Axis(fig[1, 1];
               xlabel = "t",
               ylabel = "φ",
               title  = "Order parameter (η = $η)",
               limits = (nothing, (0, 1.05)))

    lines!(ax, t, P; color = :black, linewidth = 2, label = "φ(t)")
    axislegend(ax; position = :rb)

    save(joinpath(outdir, "order_parameter_eta_$η.png"), fig, px_per_unit = 2)
end




#make_vicsek_gif(frames_x, frames_y, frames_angle, Lx, Ly, dt, save_every; filename = "vicsek.mp4")