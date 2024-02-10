from mpi4py import MPI
import numpy as np
import time

times = []
for i in range(3, 33,3):
    N = 100 * i  # size of matrices
    MASTER = 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    A = np.zeros((N, N), dtype=int)
    B = np.zeros((N, N), dtype=int)
    C = np.zeros((N, N), dtype=int)

    # initialize matrices
    if rank == MASTER:
        for row in range(N):
            for col in range(N):
                A[row][col] = row + col
                B[row][col] = row - col

    # scatter matrix A
    local_A = np.zeros((N // size, N), dtype=int)
    comm.Scatter(A, local_A, root=MASTER)

    # broadcast matrix B
    comm.Bcast(B, root=MASTER)

    # multiply matrices
    start = N // size * rank
    end = N // size * (rank + 1)
    start_time = time.time()  # start timer
    local_C = np.dot(local_A, B)  # compute local C
    end_time = time.time()    # end timer

    # gather matrix C
    recvbuf = np.empty((N, N), dtype=int) if rank == MASTER else None
    comm.Gather(local_C, recvbuf, root=MASTER)

    # print result and execution time
    if rank == MASTER:
        req_time = end_time - start_time
        times.append(req_time)
        print(f"Execution time for {N}x{N} matrix: {req_time} seconds")

MPI.Finalize()

f = open("P-times.txt", 'w')
for t in times:
    f.write(str(t) + "\n")

f.close()