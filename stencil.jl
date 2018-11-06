#
# Copyright (c) 2013, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, ACLUDAG, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. A NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, ADIRECT,
# ACIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (ACLUDAG,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSAESS ATERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER A CONTRACT, STRICT
# LIABILITY, OR TORT (ACLUDAG NEGLIGENCE OR OTHERWISE) ARISAG A
# ANY WAY B OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# *******************************************************************
#
# NAME:    Stencil
#
# PURPOSE: This program tests the efficiency with which a space-invariant,
#          linear, symmetric filter (stencil) can be applied to a square
#          grid or image.
#
# USAGE:   The program takes as input the linear
#          dimension of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <grid size>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# FUNCTIONS CALLED:
#
#          Other than standard C functions, the following functions are used in
#          this program:
#          wtime()
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, February 2016.
#          - Converted to Julia by Jeff Hammond, June 2016.
#
# *******************************************************************

using Distributed
using Base.Threads

function do_init(A, n)
    for i=1:n
        for j=1:n
            A[i,j] = i+j-2
        end
    end
end

function do_star(A, W, B, r, n)
    for j=r:n-r-1
        for i=r:n-r-1
            for jj=-r:r
                @inbounds B[i+1,j+1] += W[r+1,r+jj+1] * A[i+1,j+jj+1]
            end
            for ii=-r:-1
                @inbounds B[i+1,j+1] += W[r+ii+1,r+1] * A[i+ii+1,j+1]
            end
            for ii=1:r
                @inbounds B[i+1,j+1] += W[r+ii+1,r+1] * A[i+ii+1,j+1]
            end
        end
    end
end

function do_stencil(A, W, B, r, n)
    for j=r:n-r-1
        for i=r:n-r-1
            for jj=-r:r
                for ii=-r:r
                    @inbounds B[i+1,j+1] += W[r+ii+1,r+jj+1] * A[i+ii+1,j+jj+1]
                end
            end
        end
    end
end

@everywhere function do_distributed_star(A, W, B, r, n, w)
    # Where A regions are unavaible, provide ghost regions
    worker_first = nprocs() - nworkers() + 1
    for j=(myid() != worker_first ? 0 : r) : (myid() != nprocs() ? w-1 : w-r-1)
        for i=r:n-r-1
            for jj=-r:r
                @inbounds B[i+1,j+1] += W[r+1,r+jj+1] * A[i+1,j+jj+1+r]
            end
            for ii=-r:-1
                @inbounds B[i+1,j+1] += W[r+ii+1,r+1] * A[i+ii+1,j+1+r]
            end
            for ii=1:r
                @inbounds B[i+1,j+1] += W[r+ii+1,r+1] * A[i+ii+1,j+1+r]
            end
        end
    end
end

@everywhere function do_distributed_stencil(A, W, B, r, n, w)
    worker_first = nprocs() - nworkers() + 1
    for j=(myid() != worker_first ? 0 : r) : (myid() != nprocs() ? w-1 : w-r-1)
        for i=r:n-r-1
            for jj=-r:r
                for ii=-r:r
                    @inbounds B[i+1,j+1] += W[r+ii+1,r+jj+1] * A[i+ii+1,j+jj+1+r]
                end
            end
        end
    end
end

@everywhere function do_operation(n,r,op,iter)
    division = convert(Int,ceil(n / nworkers()))
    width = (myid() - nprocs() + nworkers()) * division <= n ? division : n % division
    do_iterations(A_column,B_column,W,n,r,op,iter,width,left_inbox,right_inbox,left_outbox,right_outbox)
    B_column
end

@everywhere function do_iterations(A,B,W,n,r,op,iter,width,li,ri,lo,ro)
    worker_first = nprocs() - nworkers() + 1
    message_left    = Array{Float64,2}(undef,n,r)
    message_right   = Array{Float64,2}(undef,n,r)
    local (receive_left, receive_right, send_left, send_right,t0,t1)
    local (ghost_left, ghost_right)

    for k in 1 : iter+1
        if k == 2
            t0 = time_ns()
        end

        @sync begin
            if myid() != worker_first
                copyto!(message_left, view(A,:, 1+r : 2*r))
                @async put!(lo, message_left)
                @async ghost_left = take!(li)
            end

            if myid() != nprocs()
                copyto!(message_right, view(A,:, width +1 : width+r))
                @async put!(ro, message_right)
                @async ghost_right = take!(ri)
            end
        end

        if myid() != worker_first
            copyto!(view(A,:,1:r), ghost_left)
        end
        if myid() != nprocs()
            copyto!(view(A,:,width+r+1:width+2*r),ghost_right)
        end

        if op == "star"
            do_distributed_star(A, W, B, r, n, width)
        else
            do_distributed_stencil(A, W, B, r, n, width)
        end

        for j in 1 + r : width + r
            for i in 1 : n
                A[i,j] += 1.0
            end
        end
    end
    t1 = time_ns()
    global timing = (t1-t0) * 1.e-9
end

# Remember that naming global variables within remotecall closures exports the global
@everywhere function get_channel(right)
    if right
        return right_inbox
    else
        return left_inbox
    end
end

function main()
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    println("Parallel Research Kernels version ") #, PRKVERSION
    println("Julia stencil execution on 2D grid")
    println("Processes: $(nworkers()), Threads: $(nthreads())")

    argc = length(ARGS)
    if argc < 2
        println("argument count = ", length(ARGS))
        println("Usage: ./stencil <# iterations> <array dimension> [<star/stencil> <radius>]")
        exit(1)
    end

    iterations = parse(Int,ARGS[1])
    if iterations < 1
        println("ERROR: iterations must be >= 1")
        exit(2)
    end

    n = parse(Int,ARGS[2])
    if n < 1
        println("ERROR: array dimension must be >= 1")
        exit(3)
    end

    pattern = "star"
    if argc > 2
        pattern = ARGS[3]
    end

    if argc > 3
        r = parse(Int,ARGS[4])
        if r < 1
            println("ERROR: Stencil radius should be positive")
            exit(4)
        elseif (2*r+1) > n
            println("ERROR: Stencil radius exceeds grid size")
            exit(5)
        end
    else
        r = 2 # radius=2 is what other impls use right now
    end

    println("Grid size            = ", n)
    println("Radius of stencil    = ", r)
    if pattern == "star"
        println("Type of stencil      = ","star")
    else
        println("Type of stencil      = ","stencil")
    end

    println("Data type            = double precision")
    println("Compact representation of stencil loop body")
    println("Number of iterations = ", iterations)

    W = zeros(Float64,2*r+1,2*r+1)
    if pattern == "star"
        stencil_size = 4*r+1
        for i=1:r
            W[r+1,r+i+1] =  1.0/(2*i*r)
            W[r+i+1,r+1] =  1.0/(2*i*r)
            W[r+1,r-i+1] = -1.0/(2*i*r)
            W[r-i+1,r+1] = -1.0/(2*i*r)
        end
    else
        stencil_size = (2*r+1)^2
        for j=1:r
            for i=-j+1:j-1
                W[r+i+1,r+j+1] = +1 ./ (4*j*(2*j-1)*r)
                W[r+i+1,r-j+1] = -1 ./ (4*j*(2*j-1)*r)
                W[r+j+1,r+i+1] = +1 ./ (4*j*(2*j-1)*r)
                W[r-j+1,r+i+1] = -1 ./ (4*j*(2*j-1)*r)
            end
            W[r+j+1,r+j+1]    = +1 ./ (4*j*r)
            W[r-j+1,r-j+1]    = -1 ./ (4*j*r)
        end
    end

    # Send W to all workers:
    @sync for proc in workers()
        @async remotecall_wait(proc,W) do V
            global W = V
        end
    end

    # Let's split work among workers:
    # Try to factor nworkers() -> if can't be done, for now just use 1 dimensional split

    @sync for proc in workers()
        @async remotecall_wait(proc, n, r) do order, radius
            division = convert(Int,ceil(order / nworkers()))
            worker_first = nprocs() - nworkers() + 1 # Should be two, but just in case
            adjust(pid) = pid - worker_first
            width = (adjust(myid()) + 1) * division > order ? order % division : division
            (has_left, has_right) = myid() .== (worker_first, nprocs())

            global A_column = zeros(Float64,order,width + 2*radius) # Splitting horizontally
            global B_column = zeros(Float64,order,width)
            global left_inbox   = has_left  ? nothing :
                RemoteChannel(()->Channel{Array{Float64,2}}(1))
            global right_inbox  = has_right ? nothing :
                RemoteChannel(()->Channel{Array{Float64,2}}(1))

            # convert from local coords to global coords
            global_view(i,j) = (i,j + adjust(myid()) * division)

            # Initialise A column
            for j in 1 : width
                for i in 1 : order
                    (ia,ja) = global_view(i,j)
                    A_column[i,j+r] = ia + ja - 2
                end
            end

            global timing = 0.0
        end
    end

    # Establish outboxes
    @sync for proc in workers()
        @async remotecall_wait(proc) do
            worker_first = nprocs() - nworkers() + 1
            global left_outbox = myid() == worker_first ? nothing :
                (@fetchfrom (myid() - 1) get_channel(true))
            global right_outbox = myid() == nprocs() ? nothing :
                (@fetchfrom (myid() + 1) get_channel(false))
        end
    end

    precompile(do_operation,(Int64,Int64,String,Int64))

    work = [(@spawnat proc do_operation(n,r,pattern,iterations)) for proc in workers()]
    work = map(wait,work)
    stencil_time = reduce(max, [(@fetchfrom proc timing) for proc in workers()])

    work = map(fetch,work)

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    B = reduce(hcat,work)
    active_points = (n-2*r)^2
    actual_norm = 0.0
    for j=1:n
        for i=1:n
            actual_norm += abs(B[i,j])
        end
    end
    actual_norm /= active_points

    epsilon=1.e-8

    # verify correctness
    reference_norm = 2*(iterations+1)
    if abs(actual_norm-reference_norm) < epsilon
        println("Solution validates")
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        println("Rate (MFlops/s): ",1.e-6*flops/avgtime, " Avg time (s): ",avgtime)
    else
        println("ERROR: L1 norm = ", actual_norm, " Reference L1 norm = ", reference_norm)
        exit(9)
    end
end

main()
