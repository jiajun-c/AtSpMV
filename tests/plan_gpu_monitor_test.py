#!/usr/bin/env python3
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from benchmark_plan_selection import recently_active_resident, shared_gpu_requires_wait, extended_baseline_minimum_bytes, fallback_tuning_minimum_bytes
from analyze_tuning_memory_bound import bound

class MockNvml:
    class NVMLError_NotFound(Exception):pass
    def __init__(self,samples):self.samples=samples
    def nvmlDeviceGetProcessUtilization(self,device,since):
        if self.samples is None:raise self.NVMLError_NotFound()
        return self.samples

samples=[SimpleNamespace(pid=1,smUtil=50,memUtil=0,timeStamp=1),
         SimpleNamespace(pid=2,smUtil=0,memUtil=0,timeStamp=2),
         SimpleNamespace(pid=3,smUtil=0,memUtil=1,timeStamp=3)]
assert recently_active_resident(MockNvml(samples),None,[2])==[]
assert [x['pid'] for x in recently_active_resident(MockNvml(samples),None,[2,3])]==[3]
assert recently_active_resident(MockNvml(None),None,[3])==[]
state={'utilization_percent':20,'free_memory_mib':4096}
assert shared_gpu_requires_wait(state,[],True,True)
assert not shared_gpu_requires_wait(state,[],False,True) # own recent utilization
assert shared_gpu_requires_wait(state,[],False,False) # no per-process monitor
assert shared_gpu_requires_wait(state,[{'pid':3}],False,True)
assert shared_gpu_requires_wait({'utilization_percent':0,'free_memory_mib':1023},[],False,True)
small={'rows':'100','columns':'1000','nnz':'1000'}
wide={'rows':'100','columns':'40000','nnz':'1000'}
assert extended_baseline_minimum_bytes(small)==12000+30000+24000+1212+8800
assert extended_baseline_minimum_bytes(wide)==12000+36000+24000+1212+8*40100
queen={'rows':'4147110','columns':'4147110','nnz':'329499284','longest_row_nnz':'81'}
assert 69*2**30 < fallback_tuning_minimum_bytes(queen) < 81559*1024*1024
assert fallback_tuning_minimum_bytes(queen)-fallback_tuning_minimum_bytes(queen,True)==6*12*329499284
outlier={'rows':'2000000','columns':'2000000','nnz':'1100000','longest_row_nnz':'1100000'}
assert fallback_tuning_minimum_bytes(outlier)==extended_baseline_minimum_bytes(outlier)
features={'dataset':'Queen_4147','rows':'4147110','cols':'4147110','nnz':'329499284',
          'long_rows':'0','short_nnz':'329499284','longest':'81'}
for i,tiles in enumerate((12206496,12204382,11167613,10678048,10419154,10296963)):
    features[f'short_tiles_{i}']=str(tiles)
assert 74.62 < bound(features)['minimum_bytes']/2**30 < 74.63
try:
    bound(dict(features,long_rows='1'))
    raise AssertionError('partitioned features must not be treated as full matrix')
except ValueError:
    pass
print('resident GPU monitor passed')
