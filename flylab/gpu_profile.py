"""NVTX stages and sampled device allocation high-water mark."""
from contextlib import contextmanager
import threading
import time


@contextmanager
def nvtx_range(label):
    try:
        import cupy as cp
        cp.cuda.nvtx.RangePush(label)
    except (ImportError,RuntimeError):
        yield;return
    try:yield
    finally:cp.cuda.nvtx.RangePop()


class DeviceMemorySampler:
    def __init__(self,interval_s=.005):
        self.interval=interval_s;self.samples=[];self.stop_event=threading.Event();self.error=None

    def __enter__(self):
        def run():
            try:
                import cupy as cp
                with cp.cuda.Device(0):
                    while not self.stop_event.is_set():
                        free,total=cp.cuda.runtime.memGetInfo()
                        self.samples.append((time.perf_counter(),int(total-free)))
                        self.stop_event.wait(self.interval)
            except Exception as exc:self.error=str(exc)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self

    def __exit__(self,*args):self.stop_event.set();self.thread.join(timeout=2.)

    def report(self):
        return dict(sampled_device_peak_bytes=max([0]+[v for _,v in self.samples]),samples=len(self.samples),
            sample_interval_s=self.interval,scope='device-wide allocated memory including other applications; sampled peak lower bound',error=self.error)


def summarize_timeline(path):
    """Read recorded CUDA allocation events and NVTX interval overlaps."""
    import sqlite3
    counts={};memory=None;stages=[]
    with sqlite3.connect(path) as c:
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names=('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_RUNTIME','NVTX_EVENTS','CUDA_GPU_MEMORY_USAGE_EVENTS','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')
        for name in names:
            if name in tables:counts[name]=c.execute('SELECT COUNT(*) FROM '+name).fetchone()[0]
        if 'CUDA_GPU_MEMORY_USAGE_EVENTS' in tables and 'ENUM_CUDA_MEM_KIND' in tables:
            kinds={r[0] for r in c.execute('SELECT * FROM ENUM_CUDA_MEM_KIND') if r[2] in ('Device','Array','Managed','Device Static','Managed Static')}
            allocated={};current=peak=0
            for pid,device,context,address,size,kind,operation in c.execute('SELECT globalPid,deviceId,contextId,address,bytes,memKind,memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start'):
                if kind not in kinds:continue
                key=(pid,device,context,address);current-=allocated.pop(key,0)
                if operation==0:allocated[key]=size;current+=size
                peak=max(peak,current)
            memory=dict(traced_cuda_allocation_peak_bytes=peak,
                scope='Profiled process CUDA device/array/managed allocations. Excludes pinned host memory and untraced driver overhead; not total physical VRAM.')
        if 'NVTX_EVENTS' in tables:
            for start,end,label in c.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE end IS NOT NULL'):
                row=dict(name=label,wall_s=(end-start)/1e9)
                for label,table in (('kernels','CUPTI_ACTIVITY_KIND_KERNEL'),('copies','CUPTI_ACTIVITY_KIND_MEMCPY'),('synchronization','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')):
                    if table not in tables:continue
                    count,duration=c.execute('SELECT COUNT(*),SUM(MIN(end,?)-MAX(start,?)) FROM '+table+' WHERE start<? AND end>?',(end,start,end,start)).fetchone()
                    row[label]=dict(count=count,summed_overlap_s=(duration or 0)/1e9)
                stages.append(row)
    return dict(timeline_tables=counts,allocation_memory=memory,stages=stages,
        duration_scope='Sums of event overlaps with CPU NVTX intervals, not exclusive stage costs or critical-path latency; overlapping kernels can sum beyond wall time.')
