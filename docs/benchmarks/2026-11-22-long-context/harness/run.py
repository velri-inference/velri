"""Three fresh processes per engine and length, with per-PID memory accounting."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from memory import run_monitored

ENGINES = {
    'velri-cpu-f32': ('cpu', 'velri', 'f32'),
    'llama-cpu-f32': ('cpu', 'llama', 'f32'),
    'velri-vulkan-f16kv': ('vulkan', 'velri', 'mixed'),
    'llama-cuda-f16kv': ('cuda', 'llama', 'mixed'),
    'llama-vulkan-f16kv': ('vulkan', 'llama', 'mixed'),
    'velri-vulkan-f32': ('vulkan', 'velri', 'f32'),
    'llama-cuda-f32': ('cuda', 'llama', 'f32'),
    'llama-vulkan-f32': ('vulkan', 'llama', 'f32'),
}
CHUNK = 1024
STEPS = 32
LENGTHS = [16384, 32768, 65536]


def run(a, name, n, block):
    with Path('/tmp/velri-gap-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        measure(a, name, n, block)


def measure(a, name, n, block):
    device, engine, profile = ENGINES[name]
    cpu = device == 'cpu'
    velri = engine == 'velri'
    mixed = profile == 'mixed'
    rec = a.out/'records'/f'{name}-{n}-{block}'
    marker = rec.with_suffix('.json')
    if marker.is_file():
        return
    prefix = a.bin/('cpu' if cpu else 'gpu')
    exe = prefix/(engine if cpu else f'velri-{profile}' if velri else f'llama-{device}')
    if not exe.is_file():
        raise FileNotFoundError(exe)
    workers = 16 if cpu else 1
    cap = n+256
    common = ['1',str(workers),'3',str(STEPS),str(n)]
    model = a.model.resolve()
    cmd = [exe, device, *common] if velri else [exe, model, *common, str(CHUNK),str(cap)]
    cmd = ['taskset','-c','0-15',*[str(x) for x in cmd]]
    forced = a.out/f'forced-{n}.txt'
    ids = ''.join(str((37*(n+j)+11)%8192)+'\n' for j in range(STEPS))
    if not forced.is_file():
        forced.write_text(ids)
    assert forced.read_text() == ids
    snapshot = a.snap/f'{rec.name}.f32'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('VELRI_','BENCH_','GGML_','NVIDIA_TF32_','CUDA_VISIBLE_DEVICES','VK_DRIVER_FILES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'))}
    env.update(BENCH_GGUF=str(model),BENCH_HISTORY='0',BENCH_CONTEXT=str(cap),BENCH_WARM_PROMPT='4096',
               VELRI_FORCE_TOKENS=str(forced.resolve()),VELRI_SPEED_LOGITS=str(snapshot.resolve()),
               VELRI_BENCH_REPLAY='1',VELRI_LLAMA_KV='f16' if mixed else 'f32',
               VELRI_LLAMA_MATH='native' if mixed else 'strict',VELRI_LLAMA_FA='on' if mixed else 'off',
               VELRI_LLAMA_DEVICE={'cpu':'CPU','vulkan':'Vulkan0','cuda':'CUDA0'}[device])
    if cpu:
        env.update(GGML_DISABLE_VULKAN='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(workers), OPENBLAS_NUM_THREADS=str(workers))
    elif not mixed and not velri and device == 'cuda':
        env.update(GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32',NVIDIA_TF32_OVERRIDE='0')
    with rec.with_suffix('.csv').open('w') as out, rec.with_suffix('.log').open('w') as log:
        print(time.strftime('%H:%M:%S'), 'start', name, n, block, flush=True)
        start = time.monotonic()
        mem = run_monitored(cmd,env,out,log,trace=rec.with_suffix('.trace.csv.gz'),mode='cpu' if cpu else 'gpu',interval_ms=10)
        wall = time.monotonic()-start
    if mem['returncode']:
        failure = dict(engine=name, prompt=n, capacity=cap, block=block, workers=workers,
                       warm_prompt=4096, warmups=3, wall_s=wall, command=cmd, **mem)
        rec.with_suffix('.failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise RuntimeError((name,n,block,mem['returncode'],rec.with_suffix('.log').read_text()[-1600:]))
    assert mem['rss_bytes']>0 and (mem['vram_bytes']==0 if cpu else mem['vram_bytes'] and mem['vram_bytes']>0), mem
    with rec.with_suffix('.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==STEPS+1, rows
    x = np.fromfile(snapshot,'<f4').reshape(STEPS+1,8192)
    assert np.isfinite(x).all()
    greedy = x.argmax(axis=1).tolist()
    for j,row in enumerate(rows):
        assert (row['phase'],int(row['rep']),int(row['step']),int(row['token'])) == ('decode' if j else 'prefill',0,j,greedy[j])
        assert float(row['us'])>0
    times = [float(o['us']) for o in rows]
    logtext = rec.with_suffix('.log').read_text()
    if velri:
        assert f'workers={workers} warmups=3 reps=1 prompt={n} chunk=1024 decode=32' in logtext
        assert 'xfers_decode=0 xfers_prefill=0' in logtext
    else:
        assert f'ctx={cap} batch=1024 ubatch=1024 steps=32 full_host_logits=8192 forced=1' in logtext
    info = dict(engine=name,prompt=n,capacity=cap,block=block,workers=workers,warm_prompt=4096,warmups=3,
        prefill_us=times[0],prefill_tps=n*1e6/times[0],
        decode_tps=STEPS*1e6/sum(times[1:]),decode_p50_us=statistics.median(times[1:]),wall_s=wall,
        command=cmd,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),greedy=greedy,
        logits_sha256=hashlib.sha256(x.tobytes()).hexdigest(),**mem)
    marker.write_text(json.dumps(info,indent=2)+'\n')
    print(time.strftime('%H:%M:%S'),'done', name, n, block, 'PP',round(info['prefill_tps'],1),'TG',round(info['decode_tps'],1),'RAM',round(mem['rss_bytes']/2**20,1),'VRAM',round(mem['vram_bytes']/2**20,1),flush=True)


def main():
    prs=argparse.ArgumentParser(description=__doc__)
    for field in ('bin','out','snap','model'):
        prs.add_argument('--'+field,type=Path,required=True)
    prs.add_argument('--engines',nargs='+',choices=ENGINES,required=True)
    prs.add_argument('--lengths',nargs='+',type=int,choices=LENGTHS,default=LENGTHS)
    prs.add_argument('--blocks',nargs='+',type=int,choices=(1,2,3),default=[1,2,3])
    a=prs.parse_args()
    (a.out/'records').mkdir(parents=True,exist_ok=True)
    a.snap.mkdir(parents=True,exist_ok=True)
    failures=[]
    for n in a.lengths:
        for block in a.blocks:
            offset=(block-1)%len(a.engines)
            order=a.engines[offset:]+a.engines[:offset]
            if block==2 and len(order)>2: order=[order[0],*order[:0:-1]]
            for name in order:
                try: run(a,name,n,block)
                except Exception as exc:
                    print(time.strftime('%H:%M:%S'),'FAILED',name,n,block,repr(exc),flush=True)
                    failures.append((name,n,block))
    if failures:
        sys.exit(str(failures))

if __name__=='__main__':
    main()
