# pure mpi
from mpi4py import MPI
import numpy as np
from multiprocessing import Process, Queue, cpu_count, shared_memory
import time

# MPI communicator
comm = MPI.COMM_WORLD
# MPI size of communicator
size = comm.Get_size()
# MPI rank of each process
rank = comm.Get_rank()

N = 10000000

def initialize_array(n):
     a=[1] * N
     b=[1] * N
     return a, b

def local_sum_func(a, b):
    for i in range(len(a)) :
        a[i] += b[i]
    return a

def compute_sum_shared (vector, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    #num_processes=2
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


def parallel(a, b, size, rank) :

    # Ensure the arrays are divided evenly among processes
    local_n = len(a) // size
    start_index = local_n * rank
    end_index = local_n * (rank + 1)
    # print(start_index, end_index)
    local_array1 = a[start_index:end_index]
    local_array2 = b[start_index:end_index]

    # Local arrays to store received data

    local_sum=local_sum_func(local_array1, local_array2)
    total_local=compute_sum_shared(np.array(local_sum), 2)
    return total_local

if __name__ == "__main__":
    # Example input arrays

    init_time0 = time.time()
    a, b = initialize_array(N)
    init_time1 = time.time()

    # Add arrays
    add_time0 = time.time()
    total = parallel(a,b, size, rank)
    #print (total)
    add_time1 = time.time()
    global_sum = comm.reduce(total, op=MPI.SUM, root=0)

    if rank==0:
        print(global_sum)
        average=global_sum/N
        print(f"Average: {average:.1f}")
        print("total time:", add_time1 - add_time0)
