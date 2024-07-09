import os
import psutil
import time
import threading
from contextlib import contextmanager
import pynvml as nvml
import numpy as np
from ..base import BeamBase
from dataclasses import dataclass


@dataclass
class PeakUsageRecords:
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    gpu_count: int


class BeamProfiler(BeamBase):
    def __init__(self, *args, percentile=99, **kwargs):

        super().__init__(*args, percentile=percentile, **kwargs)
        self.cpu_usage_values = []
        self.memory_usage_values = []
        self.gpu_memory_usage_values = []
        self.process = psutil.Process(os.getpid())
        nvml.nvmlInit()
        self.gpu_count = 0
        self.percentile = self.get_hparam('percentile', default=99)
        self.is_container = self._detect_container()

    @staticmethod
    def _detect_container():
        """
        Detect if the current process is running inside a Docker container.
        """
        try:
            with open('/proc/1/cgroup', 'rt') as f:
                return 'docker' in f.read()
        except Exception:
            return False

    def _get_all_processes(self):
        """
        Get all processes including child processes of the main process.
        """
        all_processes = [self.process]
        all_processes.extend(self.process.children(recursive=True))
        return all_processes

    def _get_host_pid(self, container_pid):
        """
        Map the container PID to the host PID.
        """
        if not self.is_container:
            return container_pid
        try:
            with open(f'/proc/{container_pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('Ngid:'):
                        pids = line.split()
                        if len(pids) > 1:
                            return int(pids[1])
        except Exception as e:
            print(f"Error getting host PID for container PID {container_pid}: {e}")
        return container_pid

    def _record_stats(self):
        try:
            # Get CPU and memory usage for the current process and its children
            total_cpu = 0
            total_memory = 0
            for proc in self._get_all_processes():
                try:
                    total_cpu += proc.cpu_percent(interval=None)
                    total_memory += proc.memory_info().rss / (1024 * 1024)  # Convert to MB
                except psutil.NoSuchProcess:
                    continue

            self.cpu_usage_values.append(total_cpu)
            self.memory_usage_values.append(total_memory)
        except psutil.NoSuchProcess:
            pass

        # Get GPU memory usage for the current process and its children
        try:
            total_gpu_memory = 0
            used_gpus = set()
            device_count = nvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                gpu_processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for process in gpu_processes:
                    host_pids = [self._get_host_pid(p.pid) for p in self._get_all_processes()]
                    if process.pid in host_pids:
                        total_gpu_memory += process.usedGpuMemory / (1024 * 1024)  # Convert to MB
                        used_gpus.add(i)

            self.gpu_memory_usage_values.append(total_gpu_memory)
            self.gpu_count = max(self.gpu_count, len(used_gpus))
        except nvml.NVMLError:
            pass

    @contextmanager
    def log(self, interval=0.1):
        self._start_monitoring(interval=interval)
        try:
            yield
        finally:
            self._stop_monitoring()
            nvml.nvmlShutdown()

    def start(self, interval=0.1):
        self._start_monitoring(interval=interval)

    def stop(self):
        self._stop_monitoring()
        nvml.nvmlShutdown()

    def _start_monitoring(self, interval=0.1):
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor, args=(interval,))
        self._monitor_thread.start()

    def _monitor(self, interval=0.1):
        while self._monitoring:
            self._record_stats()
            time.sleep(interval)  # Check every interval

    def _stop_monitoring(self):
        self._monitoring = False
        self._monitor_thread.join()

    def _calculate_percentile(self, data, percentile):
        if not data:
            return 0
        return np.percentile(data, percentile)

    @property
    def stats(self) -> PeakUsageRecords:
        # return {
        #     'cpu_usage': self._calculate_percentile(self.cpu_usage_values, self.percentile),
        #     'memory_usage': self._calculate_percentile(self.memory_usage_values, self.percentile),
        #     'gpu_memory_usage': self._calculate_percentile(self.gpu_memory_usage_values, self.percentile),
        #     'gpu_count': self.gpu_count
        # }

        return PeakUsageRecords(cpu_usage=self._calculate_percentile(self.cpu_usage_values, self.percentile),
                                memory_usage=self._calculate_percentile(self.memory_usage_values, self.percentile),
                                gpu_memory_usage=self._calculate_percentile(self.gpu_memory_usage_values,
                                                                            self.percentile),
                                gpu_count=self.gpu_count)
