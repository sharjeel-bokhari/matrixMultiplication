import time
import numpy as np

times = []
for i in range(3,33,3):
    N = 100*i  # size of matrices

    A = np.zeros((N, N), dtype=int)
    B = np.zeros((N, N), dtype=int)
    C = np.zeros((N, N), dtype=int)

    for row in range(N):
        for col in range(N):
            A[row][col] = row + col
            B[row][col] = row - col


    # multiply matrices
    start = time.time()
    np.dot(A,B)
    end = time.time()


    # calculate time difference
    print(f"Time taken for {N}x{N} matrice: {end - start} seconds\n")
    times.append(end-start)

f = open("Serial-times.txt", 'w')
for i in times:
    f.write(str(i) + "\n")
f.close()