# To Avoid the annoying warnings...
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
num_threads = len(range(35,55,2))

from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import numpy as np
import time

from main import main

# Function that is called by each thread
def thrd(thr_num, _):
    result = []
    for j in [30]:
        out = main(thr_num,j,"out_"+str(thr_num)+str(j)+".csv")
        result = result.append([thr_num, j, out[0], out[1]])
    return result

# This file was used for training multiple models at once - not very useful as RAM 
# restrictions meant that a few could be trained at any one time...
def main_2():
    pool = ThreadPool(processes=num_threads)

    thread_list = [0]*100
    thread_out = []

    for i in range(35,55,2):
        # Sleep is esential....
        time.sleep(3)
        thread_list[i] = pool.apply_async( thrd, (i,[]))

    for i in range(35,55,2):
        thread_out = thread_out.append(thread_list[i].get())

    results = np.array(thread_out)
    plt.scatter(results[:,0],results[:,1],results[:,3])

if __name__ == '__main__':
    main_2()