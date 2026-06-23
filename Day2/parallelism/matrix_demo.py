import numpy as np
import multiprocessing as mp

mat = np.random.random((8000, 3000)) # the matrix we want to process

def matrix_pool(index):
    # divide up so that we only compute one chunk of the mat.dot(mat.T) matrix
    index_start = mat.shape[0] // 10 * index
    index_end = mat.shape[0] // 10 * (index + 1)
    val  = mat[index_start:index_end].dot(mat.T)
    print("Job {0} complete".format(index))

    return val # let's return the data this time

### This entire block would need to be wrapped in `if __name__ == "__main__":`
if __name__ == "__main__":
    # Synchronous parallelization

    pool = mp.Pool(processes=4) # creae a pool with 4 worker processes


    pool_jobs = []
    for i in range(10):
        task = pool.apply_async(matrix_pool, (i,))
        pool_jobs.append(task)
        print("Created job {0}".format(i))

    for i, task in enumerate(pool_jobs):
        result = task.get()
        print("Job result {0}. First value is {1}".format(i, result.ravel()[0]))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()