import os
import time
import shlex
import subprocess
import itertools
import glob
import sys
import argparse
from predefines import epochs, processors, com_ratios

datanames = file_names = [
 'Cardiotocography',
 'Hepatitis',
 'Parkinson',
 'SpamBase',
 'WDBC',
 'WPBC',           
 'Wilt',
 'abalone',
 'amazon',
 'annthyroid',
 'arrhythmia',
 'breastw',
 'census',
 'cardio',
 'comm.and.crime',
 'cover',
 'fault',
 'glass',
 'imgseg',
 'ionosphere',
#  'letter',
 'lympho',
 'mammography',
 'mnist',
 'musk',
 'optdigits',
 'pendigits',
 'pima',
 'satellite',
 'satimage-2',
 'shuttle',
 'speech',
 'thyroid',
 'vertebral',
 'vowels',
 'wbc',
 'wine',
 'yeast',
 'backdoor',
 'fraud', 
 'campaign'
]

prototype_num = 5

seeds = [42]

def detect_device_type():
    try:
        import torch
    except Exception:
        return "cpu", 1
    if torch.cuda.is_available():
        return "cuda", max(1, torch.cuda.device_count())
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1

def available_datasets(data_dir):
    npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    names = [os.path.splitext(f)[0] for f in npz_files + mat_files]
    return set(names)


# submitting experiments in parallel to multiple devices
def run(cmds, device_type, device_id, device_cnt):
    _cur = 0

    def recycle_devices():
        running_jobs = 0
        for cid in device_id:
            if device_id[cid] is not None:
                proc = device_id[cid]
                if proc.poll() is not None:
                    device_id[cid] = None
                else:
                    running_jobs += 1
        return running_jobs

    def available_device_id():
        for cid in device_id:
            if device_id[cid] is None:
                return cid

    def submit(cmd, cid):
        env = os.environ.copy()
        if device_type == "cuda":
            env['CUDA_VISIBLE_DEVICES'] = str(int(cid) % device_cnt)
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("LOKY_MAX_CPU_COUNT", "1")

        print('Submit Job:')
        print(cmd)

        cmd_args = shlex.split(cmd)
        #log_file = open(f'log/{cmd_args[-1]}', 'w')

        proc = subprocess.Popen(cmd_args, env=env)

        device_id[cid] = proc

    while 1:
        running_jobs = recycle_devices()
        cid = available_device_id()

        if _cur >= len(cmds) and running_jobs == 0:
            break

        if cid is not None and _cur < len(cmds):
            print('{} {} available'.format(device_type.upper(), cid))
            submit(cmds[_cur], cid)
            _cur += 1

        time.sleep(5)


def start(max_parallel=None):

    cmds = []
    device_type, device_cnt = detect_device_type()
    if max_parallel is not None:
        device_cnt = max(1, min(device_cnt, max_parallel))
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    available = available_datasets(data_dir) if os.path.isdir(data_dir) else set()
    if not available:
        print(f"No datasets found in {data_dir}. Add .npz or .mat files to run jobs.")
        return
    
    # generate cmds of different experiments
    for dataname in datanames:
        if dataname not in available:
            print(f"Skip {dataname}: dataset not found in {data_dir}")
            continue
        epoch = epochs[dataname]
        processor = processors[dataname] 
        input_info_ratio, cl_ratio = com_ratios[dataname].split(',')     

        for seed in seeds:
            cmd_parts = [
                f'{shlex.quote(sys.executable)} {shlex.quote(os.path.join(os.path.dirname(__file__), "main.py"))}',
                f'--dataname {dataname}',
                f'--model_type PN',
                f'--preprocess {processor}',
                f'--diversity True',
                f'--plearn False',
                f'--input_info True', 
                f'--input_info_ratio {input_info_ratio}', 
                f'--cl True', 
                f'--cl_ratio {cl_ratio}',
                f'--prototype_num {prototype_num}',
                f'--seed {seed}',
                f'--epoch {epoch}',
                f'--device auto'
            ]
            cmd = ' '.join(cmd_parts)
            cmds.append(cmd)

    device_id = dict([(str(i), None) for i in range(device_cnt)])
    run(cmds, device_type, device_id, device_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_parallel', type=int, default=None)
    args = parser.parse_args()
    start(max_parallel=args.max_parallel)
