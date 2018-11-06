# *******************************************************************
#
# NAME:    reduce
#
# PURPOSE: This program tests the efficiency with which a collection of
#          vectors that are distributed among the threads can be added in
#          elementwise fashion. The number of vectors per thread is two,
#          so that a reduction will take place even if the code runs on
#          just a single thread.
#
# USAGE:   The program takes as input the number of threads, the length
#          of the vectors, the number of times the reduction is repeated,
#          plus, optionally, the type of reduction algorithm . The default
#          algorithm is binary tree reduction with point-to-point
#          synchronization.
#          Note that vector reduction is not currently available in C
#          in the OpenMP standard (version 2.5).
#
#          <progname> <# iterations> <vector length> [<algorithm>]
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# FUNCTIONS CALLED:
#
#          Other than OpenMP or standard C functions, the following
#          functions are used in this program:
#
#          wtime()
#          bail_out()
#
# NOTES:   The long-optimal algorithm is based on a distributed memory
#          algorithm decribed in:
#          Collective Communication; Theory, Practice, and Experience by
#          Chan, Heimlich, Purkayastha, Van de Geijn (to Appear). This
#          is a two-phase, multi-stage algorithm. In the first phase,
#          partial sums of the vectors are built by each thread locally.
#          In the second phase the partial sums are collected on the
#          master thread. In the distributed-memory algorithm the second
#          phase also has multiple stages. In the shared-memory algorithm
#          it is more efficient to let each thread write its contribution
#          into the master thread vector in a single stage.
#
# HISTORY: Written by Rohan McLure, July 2018.
#
# *******************************************************************


function do_initialize(length_vectors)
    @sync for proc in workers()
        @async remotecall_wait(proc,length_vectors) do len
            global vector = fill(1.0, len)
            global ones = fill(1.0, len)

            # Channel setup for the first rank
            first_worker = nprocs() - nworkers() + 1

            # Synchronous communication
            global inbox = RemoteChannel(()->Channel{Vector{Float64}}(0))

            global timing = 0.0
        end
    end

    # Last callout to processes to set up their outboxes
    @sync for proc in workers()
        @async remotecall_wait(proc) do
            first_worker = nprocs() - nworkers() + 1
            global reduce_channels = [(@fetchfrom proc get_channel()) for proc in sort(workers())]
        end
    end
end

@everywhere function get_channel()
    return inbox
end

@everywhere function tree_reduce(local_sum, channels, local_channel)
    first_worker = nprocs() - nworkers() + 1
    id_var = first_worker - myid() + nworkers()
    local accumulator
    if myid() != first_worker
        accumulator = copy(local_sum)
    elseif myid() == first_worker || id_var % 2 == 1
        # If you are the master than reduce into your local_sum
        accumulator = local_sum
    end

    # Indexing into the array of channel
    id_index = myid() - first_worker + 1
    k = 0
    # Order workers:        n, n-1, n-2, ..., 2, 1
    local incoming
    while id_var % 2 == 0
        incoming = take!(channels[id_index + 2^k])
        accumulator .+= incoming
        id_var ÷= 2
        k += 1
    end

    if myid() != first_worker
        put!(local_channel, accumulator)
    end
end

# Remember to use parameter passing on global reference variables to improve performance
@everywhere function do_work!(local_sum, constant_vector, channels, local_channel, iter)
    first_worker = nprocs() - nworkers() + 1
    local t0,t1
    local tree_reduce_acc = zeros(length(local_sum))
    for k in 1 : iter+1
        if k == 2
            t0 = time_ns()
        end
        local_sum .+= constant_vector

        tree_reduce(local_sum, channels, local_channel)
    end
    t1 = time_ns()
    global timing = (t1-t0) * 1.e-9
end

@everywhere function do_distributed_reduce(iterations)
    do_work!(vector, ones, reduce_channels, inbox, iterations)
end

# Similar to the MPI1 implementation, requiring communication to the first rank at each timestep
function do_reduce(iterations)
    @sync for proc in workers()
        @async remotecall_wait(do_distributed_reduce, proc, iterations)
    end
end

function do_verify(iterations, result_vector)
    result = iterations + 2.0 + (iterations*iterations + 5.0*iterations+4.0) * (nworkers()-1.0)/2.0
    ϵ = 1.e-8
    for i in 1 : length(result_vector)
        if abs(result_vector[i] - result) >= ϵ
            error("First error at i=$i; value: $(result_vector[i]); reference value: $result")
        end
    end
end


# Module rather than main function is the entrance point of Julia
function main()
    println("Parallel Research Kernels version ") #, PRKVERSION)
    println("Julia Matrix transpose: B = A^T")

    if length(ARGS) != 2
        println("argument count = ", length(ARGS))
        println("Usage: <progname> <# iterations> <vector length>")
        exit(1)
    end

    argv = map(x -> parse(Int64,x), ARGS)

    # Iterations
    iterations = argv[1]
    if iterations < 1
        println("ERROR: iterations must be >= 1")
        exit(2)
    end

    # Vector length
    vector_length = argv[2]
    if vector_length < 1
        println("ERROR: order must be >= 1")
        exit(3)
    end

    println("Vector Length            = ", vector_length)
    println("Number of iterations     = ", iterations)

    # ********************************************************************
    # ** Allocate space for the input vectors and sum vector
    # ********************************************************************

    # Populate the testing vectors
    precompile(do_initialize, (Int64,) )
    do_initialize(vector_length)

    # Precompiling the hot loop for this prk
    precompile(do_reduce, (Array{Array{Float64,1},1},))

    do_reduce(iterations)
    reduce_time = reduce(max, [(@fetchfrom proc timing) for proc in workers()])

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    first_worker = nprocs() - nworkers() + 1
    sum_vector = @fetchfrom first_worker vector

    do_verify(iterations, sum_vector)
    println("Solution validates")
    average_time = reduce_time / iterations
    rate = 1.0e-06 * (2.0*nworkers()-1)*vector_length / average_time
    println("Rate (MFlops/s): $rate  Avg time (s): $average_time")
end

main()
