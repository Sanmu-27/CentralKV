import time
import torch
import psutil
import pynvml
import json
from contextlib import contextmanager

class CentralKVProfiler:
    """
    Advanced profiler for measuring latency, memory throughput, and graph overhead.
    """
    def __init__(self, log_file="profiling_log.json"):
        self.log_file = log_file
        self.metrics = {
            "latency": [],
            "gpu_memory": [],
            "cpu_memory": [],
            "graph_overhead": []
        }
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_gpu = True
        except:
            self.has_gpu = False
            
    def measure_gpu_memory(self):
        if self.has_gpu:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return info.used / 1024**2 # MB
        return 0.0

    @contextmanager
    def timer(self, name):
        start_time = time.perf_counter()
        start_mem = self.measure_gpu_memory()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_mem = self.measure_gpu_memory()
            duration = (end_time - start_time) * 1000 # ms
            mem_delta = end_mem - start_mem
            
            entry = {
                "step": len(self.metrics["latency"]),
                "name": name,
                "duration_ms": duration,
                "memory_delta_mb": mem_delta,
                "timestamp": time.time()
            }
            
            if "graph" in name.lower():
                self.metrics["graph_overhead"].append(entry)
            else:
                self.metrics["latency"].append(entry)
                
            self.metrics["gpu_memory"].append(end_mem)
            
    def save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Profiling data saved to {self.log_file}")
            
    def summarize(self):
        latencies = [x["duration_ms"] for x in self.metrics["latency"]]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        
        graph_ovh = [x["duration_ms"] for x in self.metrics["graph_overhead"]]
        avg_graph = sum(graph_ovh) / len(graph_ovh) if graph_ovh else 0
        
        print("\n=== CentralKV Profiling Summary ===")
        print(f"Average Step Latency: {avg_lat:.4f} ms")
        print(f"Average Graph Overhead: {avg_graph:.4f} ms")
        print(f"Peak GPU Memory: {max(self.metrics['gpu_memory']) if self.metrics['gpu_memory'] else 0:.2f} MB")
        print("===================================\n")





