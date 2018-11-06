#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#*******************************************************************
#
# NAME:    transpose
#
# PURPOSE: This program measures the time for the transpose of a
#          column-major stored matrix into a row-major stored matrix.
#
# USAGE:   Program input is the matrix order and the number of times to
#          repeat the operation:
#
#          transpose <# iterations> <matrix_size>
#
#          The output consists of diagnostics to make sure the
#          transpose worked and timing statistics.
#
# HISTORY: Written by  Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#          Converted to Julia by Jeff Hammond, June 2016.
# *******************************************************************

# ********************************************************************
# read and test input parameters
# ********************************************************************

using Distributed
using Base.Threads

function do_initialize(A, order)
    for j in 1:order
        for i in 1:order
            @inbounds A[i,j] = order * (j-1) + (i-1)
        end
    end
end

function do_transpose(A, B, order)
    for j in 1:order
        for i in 1:order
            @inbounds B[i,j] += A[j,i]
            @inbounds A[j,i] += 1.0
        end
    end
end

function do_verify(B, order, iterations)
    addit = (0.5*iterations) * (iterations+1)
    abserr = 0.0
    for j in 1:order
        for i in 1:order
            temp = (order * (i-1) + (j-1)) * (iterations+1)
            @inbounds abserr = abserr + abs(B[i,j] - (temp+addit))
        end
    end
    return abserr
end

function do_parallel_initialize(order, tiling_val)
    @sync for proc in workers()
        @spawnat proc begin
            division = convert(Int,ceil(order / nworkers()))
            adjusted_id = myid() - (nprocs() - nworkers() + 1)
            block_width = (adjusted_id + 1) * division <= order ? division : order % division
            global A_column = zeros(order, block_width)
            global B_column = zeros(order, block_width)
            # Channel size should be one. Otherwise - puts and takes may not match
            global inboxes = [RemoteChannel(()->Channel{Array{Float64,2}}(1)) for proc in workers()]
            global channels = [remotecall_fetch(get_channel, proc, myid()) for proc in sort(workers())]
            global tiling   = tiling_val

            # Initialise A_column
            for j in 1 : block_width
                for i in 1 : order
                    (ja,ia) = (j + adjusted_id * division, i)
                    @inbounds A_column[i,j] = order * (ja - 1) + (ia - 1)
                end
            end

            global timing = 0.0
        end
    end
end

function do_parallel_transpose(order, iterations)
    # Create a vector of futures for work
    transpose_work = Vector{Future}(undef,nworkers())
    adjust = nprocs() - nworkers() + 1
    division = convert(Int,ceil(order / nworkers()))
    for proc in workers()
        transpose_work[proc - nprocs() + nworkers()] = remotecall(
            distributed_transpose, proc, order,iterations)
    end
    return map(fetch, transpose_work)
end

# Common pattern for returning side length. Block number is in (0 .. number of ranks - 1) representing the origin column number.
@everywhere function get_dimension(block_number, order, division)
    return (block_number + 1) * division <= order ? division : order % division
end

@everywhere function distributed_transpose(order, iterations)
    adjust = nprocs() - nworkers() + 1
    adjusted_id = myid() - adjust
    division = convert(Int, ceil(order / nworkers()))
    block_width = get_dimension(adjusted_id, order, division)

    local t0,t1

    local (incoming_block, outbound_block, t0, t1, t2, t3, t4,t)
    # Go through the phases:
    for k in 1 : iterations + 1
        if k == 2
            t0 = time_ns()
        end

        frame = division * adjusted_id + 1 : division * adjusted_id + block_width
        local_transpose(view(A_column,frame,:), view(B_column,frame,:), block_width, tiling)
        for phase in 1 : nworkers() - 1
            send_to         = (adjusted_id + nworkers() + phase) % nworkers()
            receive_from    = (adjusted_id + nworkers() - phase) % nworkers()

            incoming_height = get_dimension(receive_from, order, division)
            incoming_width  = get_dimension(adjusted_id, order, division)

            # Height of block after a transpose
            outbound_height = get_dimension(adjusted_id, order, division)
            outbound_width  = get_dimension(send_to, order, division)

            outbox          = channels[send_to+1]

            @sync begin
                @async incoming_block = take!(inboxes[receive_from + (nprocs()-nworkers())])

                @async begin
                    outbound_frame  = view(A_column, send_to * division .+ (1:outbound_width), :)
                    outbound_block  = prepare_block(outbound_frame, outbound_height, outbound_width, tiling)
                    put!(outbox, outbound_block)
                end
            end

            incoming_frame = view(B_column, receive_from * division .+ (1:incoming_height), :)
            accept_block(incoming_frame, incoming_block, incoming_width, incoming_height, tiling)
        end

    end
    t1 = time_ns()
    global timing = (t1-t0) * 1.e-9

    # Finally the function should return B_column
    return B_column
end

# Should mutate the A_Column and B_Column for each worker
@everywhere function local_transpose(A, B, block_width, tiling)
    if tiling == 0
        for j in 1 : block_width
            for i in 1 : block_width
                @inbounds B[i,j] += A[j,i]
                @inbounds A[j,i] += 1.0
            end
        end
    else
        for jt in 1 : tiling : block_width
            for it in 1 : tiling : block_width
                for j in jt : min(jt+tiling-1, block_width)
                    for i in it : min(it+tiling-1, block_width)
                        @inbounds B[i,j] += A[j,i]
                        @inbounds A[j,i] += 1.0
                    end
                end
            end
        end
    end
end

# Receive the block in my row from another process
# Sender will remote call this
@everywhere function accept_block(B, incoming_block, height, width, tiling)
    if tiling == 0
        for j in 1 : height
            for i in 1 : width
                @inbounds B[i,j] += incoming_block[i,j]
            end
        end
    else
        for jt in 1 : tiling : height
            for it in 1 : tiling : width
                for j in jt : min(jt+tiling-1, height)
                    for i in it : min(it+tiling-1, width)
                        @inbounds B[i,j] += incoming_block[i,j]
                    end
                end
            end
        end
    end
end

# Send the block in a row to process with row number. Runs on my process
@everywhere function prepare_block(A, height, width, tiling)
    outbound_block = Array{Float64}(undef,(height,width)) # Transpose dimensions
    if tiling == 0
        for j in 1 : width
            for i in 1 : height
                @inbounds outbound_block[i,j] = A[j,i]
                @inbounds A[j,i] += 1.0
            end
        end
    else
        for jt in 1 : tiling : width
            for it in 1 : tiling : height
                for j in jt : min(jt+tiling-1, width)
                    for i in it : min(it+tiling-1, height)
                        @inbounds outbound_block[i,j] = A[j,i]
                        @inbounds A[j,i] += 1.0
                    end
                end
            end
        end
    end
    return outbound_block
end

@everywhere function get_channel(id)
    return inboxes[id - (nprocs() - nworkers())]
end

function main()
    println("Parallel Research Kernels version ") #, PRKVERSION)
    println("Julia Matrix transpose: B = A^T")
    println("Processes: $(nworkers()), Threads: $(nthreads())")

    if !(2 <= length(ARGS) <= 3)
        println("argument count = ", length(ARGS))
        println("Usage: ./transpose <# iterations> <matrix order>")
        exit(1)
    end

    argv = map(x->parse(Int64,x),ARGS)

    # iterations
    iterations = argv[1]
    if iterations < 1
        println("ERROR: iterations must be >= 1")
        exit(2)
    end

    # matrix order
    order = argv[2]
    if order < 1
        println("ERROR: order must be >= 1")
        exit(3)
    end

    tiling = 0
    if length(ARGS) == 3
        tiling = argv[3]
        if tiling < 0
            println("ERROR: tiling must be >= 0")
            exit(4)
        end
    end

    println("Order                    = ", order)
    println("Number of iterations     = ", iterations)
    println("Tiling                   = ", tiling)

    # ********************************************************************
    # ** Prepare the local components.
    # ********************************************************************

    precompile(do_parallel_initialize, (Int64,Int64))
    do_parallel_initialize(order, tiling)

    # ********************************************************************
    # ** Do the distributed transpose.
    # ********************************************************************

    precompile(do_parallel_transpose, (Int64, Int64))
    @everywhere workers() precompile(distributed_transpose, (Int64, Int64))

    transpose_work = do_parallel_transpose(order, iterations)
    trans_time = reduce(max, [(@fetchfrom proc timing) for proc in workers()])

    # May sequentially fetch the results of remotecalls -> already kicked them off.
    B = reduce(hcat, transpose_work)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    precompile(do_verify, (Array{Float64,2}, Int64, Int64))
    abserr = do_verify(B, order, iterations)

    epsilon=1.e-8
    nbytes = 2 * order^2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if abserr < epsilon
        println("Solution validates")
        avgtime = trans_time/iterations
        println("Rate (MB/s): ",1.e-6*nbytes/avgtime, " Avg time (s): ", avgtime)
    else
        println("error ",abserr, " exceeds threshold ",epsilon)
        println("ERROR: solution did not validate")
        exit(1)
    end
end

main()
