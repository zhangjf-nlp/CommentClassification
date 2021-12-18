import os
import sys
import time
import datetime
import numpy as np

def dynamic_cuda_allocation():
    tmp_file_name = f"tmp-{time.time()}"
    os.system(f'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{tmp_file_name}')
    memory_gpu = [int(x.split()[2]) for x in open(tmp_file_name,'r').readlines()]
    os.system(f'rm {tmp_file_name}')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))

def block_until_cuda_memory_free(required_mem, interval=30):
    start_time = time.time()
    def get_available_mem():
        tmp_file_name = f"tmp-{time.time()}"
        os.system(f'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{tmp_file_name}')
        memory_gpu = [int(x.split()[2]) for x in open(tmp_file_name,'r').readlines()]
        os.system(f'rm {tmp_file_name}')
        return max(memory_gpu)
    available_mem = get_available_mem()
    while available_mem < required_mem:
        blocked_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
        print(f"{time.ctime()} \t {available_mem} MiB < {required_mem} MiB : blocked for {blocked_time}", end="\r")
        time.sleep(interval)
        available_mem = get_available_mem()
    blocked_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print(f"{time.ctime()} \t {available_mem} MiB >= {required_mem} MiB : passed after {blocked_time}")