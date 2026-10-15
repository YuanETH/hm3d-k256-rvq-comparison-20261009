"""Run independent experiment jobs on explicitly selected GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('stage', choices=('train', 'validation', 'test'))
p.add_argument('--cache', required=True)
p.add_argument('--official', required=True)
p.add_argument('--out', type=Path, default=Path('results'))
p.add_argument('--gpus', default='0,1,4,5,6,7')
p.add_argument('--names')
args = p.parse_args()
definitions = json.loads((args.out / 'experiments.json').read_text())
names = list(definitions['models' if args.stage == 'train' else 'variants'])
if args.names:
    names = args.names.split(',')
available = queue.Queue()
for gpu in args.gpus.split(','):
    available.put(gpu)
logdir = args.out / 'logs'
logdir.mkdir(exist_ok=True)


def run(name):
    gpu = available.get()
    try:
        command = [sys.executable, 'experiment.py', 'train' if args.stage == 'train' else 'evaluate',
                   '--name', name, '--cache', args.cache, '--official', args.official,
                   '--out', str(args.out), '--device', 'cuda:0']
        if args.stage != 'train':
            command.extend(['--split', args.stage])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        path = logdir / f'{args.stage}-{name}.log'
        with path.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        return name, result.returncode, path.read_text().splitlines()[-2:]
    finally:
        available.put(gpu)


failed = []
with ThreadPoolExecutor(max_workers=available.qsize()) as pool:
    futures = [pool.submit(run, name) for name in names]
    for index, future in enumerate(as_completed(futures), 1):
        name, code, tail = future.result()
        print(f'{index}/{len(names)} {name} exit={code} ' + ' | '.join(tail), flush=True)
        if code:
            failed.append(name)
if failed:
    raise RuntimeError('failed jobs: ' + ', '.join(failed))
