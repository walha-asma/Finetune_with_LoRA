import time
import threading
import torch
import gc

from .metrics import ResourceSnapshot, ResourceMetrics

# Import monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Install with: pip install psutil")

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    PYNVML_AVAILABLE = False


class ResourceMonitor:
    """
    Monitor system resources in background thread
    
    Usage:
        with ResourceMonitor() as monitor:
            # your code here
            pass
        metrics = monitor.get_metrics()
        metrics.print_summary()
        metrics.plot('usage.png')
    """
    
    def __init__(self, sample_rate_hz: float = 10.0, gpu_index: int = 0):
        """
        Args:
            sample_rate_hz: Sampling frequency (default 10 Hz)
            gpu_index: GPU device index
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil required. Run: pip install psutil")
        
        self.sample_interval = 1.0 / sample_rate_hz
        self.gpu_index = gpu_index
        
        self._data = {
            "time": [],
            "vram_used_mb": [],
            "vram_total_mb": [],
            "ram_used_mb": [],
            "ram_total_mb": [],
            "gpu_util": [],
            "cpu_util": [],
            "power_watts": []
        }
        
        self._thread = None
        self._running = [False]  # Use list so thread can see changes
        self._start_time = None
        self.process = psutil.Process()
        
        # Initialize GPU if available
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except:
                pass
    
    def _sample(self):
        """Take one sample of all resources"""
        current_time = time.time() - self._start_time
        
        # CPU and RAM
        cpu_util = self.process.cpu_percent(interval=0)
        ram = psutil.virtual_memory()
        ram_used_mb = ram.used / (1024 ** 2)
        ram_total_mb = ram.total / (1024 ** 2)
        
        # GPU metrics
        vram_used_mb = 0
        vram_total_mb = 0
        gpu_util = 0
        power_watts = None
        
        if self.gpu_handle:
            try:
                # VRAM via pynvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_used_mb = mem_info.used / (1024 ** 2)
                vram_total_mb = mem_info.total / (1024 ** 2)
                
                # GPU utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = float(util_info.gpu)
                
                # Power
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_watts = power_mw / 1000.0
                except:
                    pass
            except:
                pass
        elif torch.cuda.is_available():
            # Fallback to torch if pynvml not available
            try:
                vram_used_mb = torch.cuda.memory_allocated(self.gpu_index) / (1024 ** 2)
                vram_total_mb = torch.cuda.get_device_properties(self.gpu_index).total_memory / (1024 ** 2)
            except:
                pass
        
        self._data["time"].append(current_time)
        self._data["vram_used_mb"].append(vram_used_mb)
        self._data["vram_total_mb"].append(vram_total_mb)
        self._data["ram_used_mb"].append(ram_used_mb)
        self._data["ram_total_mb"].append(ram_total_mb)
        self._data["gpu_util"].append(gpu_util)
        self._data["cpu_util"].append(cpu_util)
        self._data["power_watts"].append(power_watts)
    
    def _monitoring_loop(self):
        """Background thread loop"""
        while self._running[0]:
            self._sample()
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring"""
        self._running[0] = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> ResourceMetrics:
        """Stop monitoring and return metrics"""
        self._running[0] = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.get_metrics()
    
    def get_metrics(self) -> ResourceMetrics:
        """Compute and return metrics"""
        return ResourceMetrics(self._data, self._start_time)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def cleanup_gpu():
    """Cleanup GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.5)

