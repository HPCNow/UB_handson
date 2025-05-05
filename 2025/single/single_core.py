import numpy as np
import time

N = 10000000

def initialize_array(n):
    a = np.ones(n)
    b = np.ones(n)
    return a, b

def add_arrays(a, b):
    for i in range(len(a)):
        a[i] += b[i]
    return a

def compute_sum(a):
    total = 0.0
    for i in range(len(a)):
        total += a[i]
    return total

if __name__ == "__main__":
    # Initialize
    start_time = time.time()
    init_time0 = time.time()
    a, b = initialize_array(N)
    init_time1 = time.time()
    print("Initialize arrays time:", init_time1 - init_time0)

    # Add arrays
    add_time0 = time.time()
    a = add_arrays(a, b)
    add_time1 = time.time()
    print("Add arrays time:", add_time1 - add_time0)

    # Compute average
    av_time0 = time.time()
    total = compute_sum(a)
    average = total / N
    av_time1 = time.time()
    print("Average result time:", av_time1 - av_time0)
    print("Average:", average)
    end_time=time.time()

    # Total running time
    print('Total running time:', end_time - start_time)

