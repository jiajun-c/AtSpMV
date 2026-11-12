"""Query one GPU directly; avoid spawning nvidia-smi in every polling cycle."""
from benchmark_dsm_atbsr_full import apps


class GpuProcessMonitor:
    def __init__(self, uuid):
        self.uuid = uuid
        try:
            import pynvml
        except ImportError:
            self.backend = 'nvidia-smi'
            self.nvml = None
        else:
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.device = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self.backend = 'nvml'

    def __call__(self):
        if self.nvml is None:
            return apps(self.uuid)
        return [{'pid': int(process.pid)} for process in
                self.nvml.nvmlDeviceGetComputeRunningProcesses(self.device)]
