import os,subprocess,time,json,hashlib,signal
from pathlib import Path
import pynvml
root=Path('/staff/chengjiajun/workspace/AtSpmv')
out=root/'results/model/hybrid-time-model/shared-tile-stats-h100/early-abba'
exe={'previous':root/'results/model/hybrid-time-model/short-row-cache-h100/provenance/spmv_hybrid_validation','current':root/'results/model/hybrid-time-model/shared-tile-stats-h100/provenance/spmv_hybrid_validation'}
matrices=['a0nsdsil','shermanACb','freeFlyingRobot_16']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3';pynvml.nvmlInit();gpu=pynvml.nvmlDeviceGetHandleByIndex(3)
def apps():
    return {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta={'gpu_uuid':pynvml.nvmlDeviceGetUUID(gpu),'process_order':['previous','current','current','previous'],'seed':20261028,'trials':9,'calls_per_graph':8,'method':'Previously selected examples; ABBA independent processes, two X vectors and Graph correctness checked by executable. Poll NVML for other compute processes every 250 ms. This does not exclude missed interference, clock shifts or allocation-history differences.','executables':{k:sha(v) for k,v in exe.items()},'matrix_sha256':{n:sha(Path('/staff/chengjiajun/workspace/database/suitsparse')/n/(n+'.mtx')) for n in matrices}}
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
for name in matrices:
    m=Path('/staff/chengjiajun/workspace/database/suitsparse')/name/(name+'.mtx')
    for i,kind in enumerate(meta['process_order']):
        foreign=apps()
        if foreign:raise RuntimeError(f'GPU busy before test: {foreign}')
        with (out/f'{name}-{i}-{kind}.csv.tmp').open('w') as f, (out/f'{name}-{i}-{kind}.log.tmp').open('w') as e:
            proc=subprocess.Popen([str(exe[kind]),str(m),str(out/'config.csv'),'20261028','9'],env=env,stdout=f,stderr=e,start_new_session=True)
            while proc.poll() is None:
                time.sleep(.25)
                other=apps()-{proc.pid}
                if other:
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait()
                    raise RuntimeError(f'GPU interference {other}; uncommitted files retained')
            if proc.returncode:raise RuntimeError(f'{name} {kind} returned {proc.returncode}')
        for suffix in ('csv','log'):(out/f'{name}-{i}-{kind}.{suffix}.tmp').replace(out/f'{name}-{i}-{kind}.{suffix}')
        print(name,i,kind,flush=True)
