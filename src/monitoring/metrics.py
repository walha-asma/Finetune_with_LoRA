import time
import pandas as pd


class ResourceSnapshot:
    """Single measurement (not used in simplified version, kept for compatibility)"""
    pass


class ResourceMetrics:
    """Resource usage metrics with export and plotting"""
    
    def __init__(self, data, start_time):
        """
        Args:
            data: Dict with lists of measurements
            start_time: When monitoring started
        """
        self.data = data
        self.start_time = start_time
        self.duration_seconds = data["time"][-1] if data["time"] else 0
        self.num_samples = len(data["time"])
        
        # Compute statistics
        if self.num_samples > 0:
            self.vram_mean_mb = sum(data["vram_used_mb"]) / self.num_samples
            self.vram_max_mb = max(data["vram_used_mb"])
            self.vram_min_mb = min(data["vram_used_mb"])
            
            self.ram_mean_mb = sum(data["ram_used_mb"]) / self.num_samples
            self.ram_max_mb = max(data["ram_used_mb"])
            self.ram_min_mb = min(data["ram_used_mb"])
            
            self.gpu_util_mean = sum(data["gpu_util"]) / self.num_samples
            self.gpu_util_max = max(data["gpu_util"])
            
            self.cpu_util_mean = sum(data["cpu_util"]) / self.num_samples
            self.cpu_util_max = max(data["cpu_util"])
            
            # Power (optional)
            power_values = [p for p in data["power_watts"] if p is not None]
            if power_values:
                self.power_mean_watts = sum(power_values) / len(power_values)
                self.power_max_watts = max(power_values)
                self.power_total_joules = self.power_mean_watts * self.duration_seconds
            else:
                self.power_mean_watts = None
                self.power_max_watts = None
                self.power_total_joules = None
        else:
            self.vram_mean_mb = self.vram_max_mb = self.vram_min_mb = 0
            self.ram_mean_mb = self.ram_max_mb = self.ram_min_mb = 0
            self.gpu_util_mean = self.gpu_util_max = 0
            self.cpu_util_mean = self.cpu_util_max = 0
            self.power_mean_watts = self.power_max_watts = self.power_total_joules = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame(self.data)
        if len(df) > 0:
            df = df.set_index('time')
        return df
    
    def save_csv(self, filepath: str):
        """Save to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath)
        print(f"Saved metrics to: {filepath}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 60)
        print(f"Duration: {self.duration_seconds:.2f}s")
        print(f"Samples: {self.num_samples} ({self.num_samples/self.duration_seconds:.1f} Hz)")
        print()
        print(f"VRAM:  Mean: {self.vram_mean_mb:.1f} MB  |  Peak: {self.vram_max_mb:.1f} MB")
        print(f"RAM:   Mean: {self.ram_mean_mb:.1f} MB  |  Peak: {self.ram_max_mb:.1f} MB")
        print(f"GPU:   Mean: {self.gpu_util_mean:.1f}%  |  Peak: {self.gpu_util_max:.1f}%")
        print(f"CPU:   Mean: {self.cpu_util_mean:.1f}%  |  Peak: {self.cpu_util_max:.1f}%")
        
        if self.power_mean_watts:
            print(f"Power: Mean: {self.power_mean_watts:.1f} W  |  Peak: {self.power_max_watts:.1f} W")
            print(f"Energy: {self.power_total_joules:.1f} J ({self.power_total_joules/3600:.4f} Wh)")
        
        print("=" * 60)
    
    def plot(self, save_path: str = None):
        """Plot resource usage"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        df = self.to_dataframe()
        if df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Resource Usage Over Time', fontsize=16, fontweight='bold')
        
        # VRAM
        axes[0, 0].plot(df.index, df['vram_used_mb'], color='red', linewidth=2)
        axes[0, 0].fill_between(df.index, 0, df['vram_used_mb'], alpha=0.3, color='red')
        axes[0, 0].set_ylabel('VRAM (MB)')
        axes[0, 0].set_title(f'GPU Memory (Peak: {self.vram_max_mb:.0f} MB)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RAM
        axes[0, 1].plot(df.index, df['ram_used_mb'], color='blue', linewidth=2)
        axes[0, 1].fill_between(df.index, 0, df['ram_used_mb'], alpha=0.3, color='blue')
        axes[0, 1].set_ylabel('RAM (MB)')
        axes[0, 1].set_title(f'System Memory (Peak: {self.ram_max_mb:.0f} MB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU
        axes[1, 0].plot(df.index, df['gpu_util'], color='green', linewidth=2)
        axes[1, 0].fill_between(df.index, 0, df['gpu_util'], alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('GPU Utilization (%)')
        axes[1, 0].set_title(f'GPU Usage (Peak: {self.gpu_util_max:.0f}%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)
        
        # CPU
        axes[1, 1].plot(df.index, df['cpu_util'], color='orange', linewidth=2)
        axes[1, 1].fill_between(df.index, 0, df['cpu_util'], alpha=0.3, color='orange')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('CPU Utilization (%)')
        axes[1, 1].set_title(f'CPU Usage (Peak: {self.cpu_util_max:.0f}%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()

