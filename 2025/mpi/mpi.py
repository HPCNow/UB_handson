# pure mpi
from mpi4py import MPI
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
    total_local=0.0
    for i in range(len(local_sum)) :
        total_local += local_sum[i]
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
        average=global_sum/N
        print(f"Average: {average:.1f}")
        print("total time:", add_time1 - add_time0)
