import numpy as np
import time
from multiprocessing import Process, Queue, cpu_count, shared_memory
import argparse

parser = argparse.ArgumentParser(description="Arguments for parallelization")
parser.add_argument("--numproc", type=int, default=2 , help="Number of processes")
args = parser.parse_args()

num_processes=args.numproc

N = 10000000

def initialize_array(n):
    a = np.ones(n)
    b = np.ones(n)
    return a, b

def add_arrays(arr1, arr2, num_processes=None):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be the same length")

    if num_processes is None:
        num_processes = cpu_count()

    n = len(arr1)
    chunk_size = n // num_processes

    # Create shared memory blocks
    shm_a = shared_memory.SharedMemory(create=True, size=arr1.nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=arr2.nbytes)
    shm_result = shared_memory.SharedMemory(create=True, size=arr1.nbytes)

    # Create numpy arrays backed by shared memory
    a_shared = np.ndarray(arr1.shape, dtype=arr1.dtype, buffer=shm_a.buf)
    b_shared = np.ndarray(arr2.shape, dtype=arr2.dtype, buffer=shm_b.buf)
    result_shared = np.ndarray(arr1.shape, dtype=arr1.dtype, buffer=shm_result.buf)

    # Copy data to shared memory
    np.copyto(a_shared, arr1)
    np.copyto(b_shared, arr2)

    def worker(start, end):
        a = np.ndarray(arr1.shape, dtype=arr1.dtype, buffer=shm_a.buf)[start:end]
        b = np.ndarray(arr2.shape, dtype=arr2.dtype, buffer=shm_b.buf)[start:end]
        r = np.ndarray(arr1.shape, dtype=arr1.dtype, buffer=shm_result.buf)[start:end]
        r[:] = a + b

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = n if i == num_processes - 1 else (i + 1) * chunk_size
        p = Process(target=worker, args=(start, end))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Retrieve the result
    result = np.ndarray(arr1.shape, dtype=arr1.dtype, buffer=shm_result.buf).copy()

    # Clean up shared memory
    shm_a.close(); shm_a.unlink()
    shm_b.close(); shm_b.unlink()
    shm_result.close(); shm_result.unlink()

    return result

def compute_sum_shared(vector, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    n = len(vector)
    chunk_size = n // num_processes

    # Create shared memory block
    shm = shared_memory.SharedMemory(create=True, size=vector.nbytes)
    shm_vector = np.ndarray(vector.shape, dtype=vector.dtype, buffer=shm.buf)
    np.copyto(shm_vector, vector)

    q = Queue()

    def worker(start, end, shape, dtype, shm_name):
        shm_local = shared_memory.SharedMemory(name=shm_name)
        local_vector = np.ndarray(shape, dtype=dtype, buffer=shm_local.buf)
        q.put(np.sum(local_vector[start:end]))
        shm_local.close()

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = n if i == num_processes - 1 else (i + 1) * chunk_size
        p = Process(target=worker, args=(
            start, end, vector.shape, vector.dtype, shm.name
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    result = sum(q.get() for _ in processes)

    shm.close()
    shm.unlink()

    return result

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
    total = compute_sum_shared(a)
    average = total / N
    av_time1 = time.time()
    print("Average result time:", av_time1 - av_time0)
    print("Average:", average)
    end_time=time.time()

    # Total running time
    print('Total running time:', end_time - start_time)

