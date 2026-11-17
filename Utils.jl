module Utils

using Random

export positions_lattice_square, positions_lattice_triangular, positions_random, rescale_periodic!
export cell_grid, build_cell_list!, check_cell_list
export build_verlet_list!, needs_rebuild, check_verlet_list, msd
export wca_cutoff, U_WCA, force_over_r_WCA, compute_forces_WCA!, compute_angle_avg!, compute_order_parameter,compute_adjacency_matrix

using Random


function positions_lattice_square(n; Lx = 1.0, Ly = 1.0)
    
    packing_fraction = get_packing_fraction(n, Lx, Ly)

    if packing_fraction > 0.9
        error("Packing fraction = $(round(packing_fraction, digits=3)) is too high: particles overlap. Increase Lx, Ly or reduce n.")
    end
    
    nx = ceil(Int, sqrt(n))      # columns
    ny = ceil(Int, n / nx)       # rows needed so that nx*ny ≥ n
    ax = Lx / nx                 # spacing in x
    ay = Ly / ny                 # spacing in y

    x = zeros(n)
    y = zeros(n)

    aux = 1

    for j in 1:ny
        for i in 1:nx
            x[aux] = (j - 0.5) * ax     # cell centers
            y[aux] = (i - 0.5) * ay

            aux += 1

            if aux > n
                rescale_periodic!(x, Lx)
                rescale_periodic!(y, Ly)              # stop after placing n particles
                return x, y
            end
        end
    end

    rescale_periodic!(x, Lx)
    rescale_periodic!(y, Ly)
    return x, y
end



function positions_lattice_triangular(n; Lx = 1.0, Ly = 1.0)

    packing_fraction = get_packing_fraction(n, Lx, Ly)

    if packing_fraction > 0.9
        error("Packing fraction = $(round(packing_fraction, digits=3)) is too high: particles overlap. Increase Lx, Ly or reduce n.")
    end

    nx = ceil(Int, sqrt(n))      # columns
    ny = ceil(Int, n / nx)       # rows needed so that nx*ny ≥ n

    a  = Lx / nx                 # nearest-neighbor distance, set by Lx
    ay = a * sqrt(3) / 2         # distance between rows

    if a < 1.0
        @warn "Lattice spacing a = $(round(a, digits=3)) is smaller than σ = 1: particles overlap."
    end

    x = zeros(n)
    y = zeros(n)

    aux = 1

    for j in 1:ny
        shift = 0.5 * mod(j - 1, 2)              # 0 on even rows, 0.5 on odd rows

        for i in 1:nx
            x[aux] = (i - 0.1 + shift) * a      # margins of a/4 on both sides
            y[aux] = (j - 0.1) * ay

            aux += 1

            if aux > n

                rescale_periodic!(x, Lx)
                rescale_periodic!(y, Ly)

                return x, y
            end
        end
    end


    rescale_periodic!(x, Ly)
    rescale_periodic!(y, Ly)

    return x, y
end

function positions_random(n; Lx = 1.0, Ly = 1.0)

    x = zeros(n)
    y = zeros(n)

    for i in 1:n
        x[i] = rand()*Lx
        y[i] = rand()*Ly
    end

    return x, y

end 


function get_packing_fraction(n, Lx, Ly; radius = 0.5)

    A = Lx * Ly
    packing_fraction = (n * π* radius^2)/A
    return packing_fraction

end


function rescale_periodic!(r::AbstractVector, limit)
    @. r = r - floor(r/limit) * limit
    return r
end


function cell_grid(Lx, Ly, r_list)
    Mx = floor(Int, Lx / r_list)
    My = floor(Int, Ly / r_list)
    (Mx >= 3 && My >= 3) || error("Need at least 3 cells per side: Mx = $Mx, My = $My. Increase the box or reduce r_list.")
    return Mx, My
end

# fills head and next in place; positions must be wrapped into [0, L)
function build_cell_list!(head, next, x, y, Lx, Ly, Mx, My)
    cx = Lx / Mx
    cy = Ly / My

    fill!(head, 0)                                  # reset: stale entries are the classic bug

    for i in eachindex(x)
        ix = clamp(floor(Int, x[i] / cx) + 1, 1, Mx)
        iy = clamp(floor(Int, y[i] / cy) + 1, 1, My)
        c  = ix + (iy - 1) * Mx                     # linear cell index, 1:Mx*My

        next[i] = head[c]                           # push particle i onto the chain of cell c
        head[c] = i
    end

    return head, next
end

# sanity check: every particle appears exactly once across all chains
function check_cell_list(head, next, n)
    seen = zeros(Int, n)
    for c in eachindex(head)
        i = head[c]
        while i != 0
            seen[i] += 1
            i = next[i]
        end
    end
    return all(==(1), seen)
end


# half Verlet list: neighbors[i] holds every j > i with |r_i - r_j| < r_list (minimum image)
# built from the cell list, so it only looks at the 3x3 block of cells around each particle
function build_verlet_list!(neighbors, head, next, x, y, Lx, Ly, Mx, My, r_list)
    build_cell_list!(head, next, x, y, Lx, Ly, Mx, My)

    cx = Lx / Mx
    cy = Ly / My
    r_list2 = r_list^2

    foreach(empty!, neighbors)                      # keeps capacity: no allocations after the first build

    for i in eachindex(x)
        ix = clamp(floor(Int, x[i] / cx) + 1, 1, Mx)
        iy = clamp(floor(Int, y[i] / cy) + 1, 1, My)

        for diy in -1:1, dix in -1:1
            jx = mod1(ix + dix, Mx)                 # periodic neighbor cell
            jy = mod1(iy + diy, My)
            j  = head[jx + (jy - 1) * Mx]

            while j != 0
                if j > i                            # each pair stored once
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dx -= Lx * round(dx / Lx)       # minimum image
                    dy -= Ly * round(dy / Ly)

                    if dx^2 + dy^2 < r_list2
                        push!(neighbors[i], j)
                    end
                end
                j = next[j]
            end
        end
    end

    return neighbors
end

# true when some particle moved more than skin/2 since the last build (x0, y0 = positions at that build)
function needs_rebuild(x, y, x0, y0, skin, Lx, Ly)
    max_d2 = (skin / 2)^2

    for i in eachindex(x)
        dx = x[i] - x0[i]
        dy = y[i] - y0[i]
        dx -= Lx * round(dx / Lx)                   # positions are wrapped, so use minimum image
        dy -= Ly * round(dy / Ly)

        if dx^2 + dy^2 > max_d2
            return true
        end
    end

    return false
end

# sanity check: compare against the O(n^2) brute force list
function check_verlet_list(neighbors, x, y, Lx, Ly, r_list)
    n = length(x)
    for i in 1:n, j in i+1:n
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        dx -= Lx * round(dx / Lx)
        dy -= Ly * round(dy / Ly)

        if (dx^2 + dy^2 < r_list^2) != (j in neighbors[i])
            return false
        end
    end
    return true
end


wca_cutoff(σ = 1.0) = 2^(1/6) * σ

# WCA: U(r) = 4ϵ[(σ/r)^12 - (σ/r)^6] + ϵ for r < 2^(1/6)σ, zero beyond
function U_WCA(r2; ϵ = 1.0, σ = 1.0)
    if r2 >= wca_cutoff(σ)^2
        return zero(r2)
    end

    s6 = (σ^2 / r2)^3                               # (σ/r)^6
    return 4ϵ * (s6^2 - s6) + ϵ
end

# -(1/r) dU/dr, so the force on i from j is F_i = f * (r_i - r_j)
function force_over_r_WCA(r2; ϵ = 1.0, σ = 1.0)
    if r2 >= wca_cutoff(σ)^2
        return zero(r2)
    end

    s6 = (σ^2 / r2)^3
    return 24ϵ * (2s6^2 - s6) / r2
end

# fills fx, fy with the WCA forces, using the half Verlet list (each pair once, Newton's third law)
function compute_forces_WCA!(fx, fy, neighbors, x, y, Lx, Ly; ϵ = 1.0, σ = 1.0)
    fill!(fx, 0.0)
    fill!(fy, 0.0)

    for i in eachindex(x)
        for j in neighbors[i]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dx -= Lx * round(dx / Lx)               # minimum image
            dy -= Ly * round(dy / Ly)

            f = force_over_r_WCA(dx^2 + dy^2; ϵ = ϵ, σ = σ)

            fx[i] += f * dx
            fy[i] += f * dy
            fx[j] -= f * dx
            fy[j] -= f * dy
        end
    end

    return fx, fy
end

# z, acc: ComplexF64 buffers of length n (unit headings and their neighborhood sums)
function compute_angle_avg!(f_angle, z, acc, neighbors, x, y, θ, Lx, Ly; r_0 = 1.0)

    r_02 = r_0^2

    @. z = cis(θ)                                   # one sincos per particle, not one per pair
    acc .= z                                        # each particle counts itself

    @inbounds for i in eachindex(θ)
        xi, yi = x[i], y[i]
        ai, zi = acc[i], z[i]

        for j in neighbors[i]
            dx = xi - x[j]
            dy = yi - y[j]
            dx -= Lx * round(dx / Lx)               # minimum image
            dy -= Ly * round(dy / Ly)

            w = ifelse(dx^2 + dy^2 < r_02, 1.0, 0.0)  # branchless: ~40% of list pairs lie in the skin

            ai     += w * z[j]                      # half list: each pair is stored once,
            acc[j] += w * zi                        # so credit it to both i and j
        end
        acc[i] = ai
    end

    @. f_angle = Base.angle(acc)
    return f_angle
end


function compute_adjacency_matrix(x, y, neighbors, Lx, Ly; r_0 = 1.0)

    n    = length(x)
    r_02 = r_0^2

    A = zeros(Int8, n, n)

    for i in eachindex(x)
        xi, yi = x[i], y[i]
        for j in neighbors[i]

            dx = xi - x[j]
            dy = yi - y[j]
            dx -= Lx * round(dx / Lx)               # minimum image
            dy -= Ly * round(dy / Ly)

            r2 = dx^2 + dy^2

            if r2 < r_02
                A[i, j] = 1
                A[j, i] = 1
            end
        end
    end

    return A
end


function msd(x, y, dt, saving_period)
    frames = length(x)
    time_conversion = dt * saving_period

    lags = 0:frames-1
    msd_values = zeros(frames)

    for lag in 1:frames-1                  # time separation, in saved frames
        acc = 0.0
        for i in 1:frames-lag              # every possible time origin
            acc += (x[i+lag] - x[i])^2 + (y[i+lag] - y[i])^2
        end
        msd_values[lag+1] = acc / (frames - lag)
    end

    return lags .* time_conversion, msd_values
end

function compute_order_parameter(angles)
    
    n = length(angles)
    acc = 0.0 + 0.0im
    for θ in angles
        acc += cis(θ)
    end

    return abs(acc) / n
end




end