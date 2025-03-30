import ctypes
from ctypes import byref, c_ulonglong
import time

class HPETTimer:
    """High Precision Event Timer for Windows."""
    
    def __init__(self):
        self.qpc_freq = c_ulonglong(0)
        self.available = False
        
        try:
            # Try to initialize the high-precision timer
            if ctypes.windll.kernel32.QueryPerformanceFrequency(byref(self.qpc_freq)):
                self.available = True
        except Exception:
            self.available = False
            
    def is_available(self):
        """Check if HPET is available."""
        return self.available
            
    def get_time(self):
        """Get the current time in seconds using high precision timer."""
        if not self.available:
            return time.perf_counter()
            
        current_time = c_ulonglong(0)
        ctypes.windll.kernel32.QueryPerformanceCounter(byref(current_time))
        return current_time.value / self.qpc_freq.value
        
    def sleep(self, seconds):
        """Sleep for the specified number of seconds with high precision."""
        if not self.available:
            time.sleep(seconds)
            return
            
        start_time = self.get_time()
        end_time = start_time + seconds
        
        # For very short sleeps, busy-wait
        while self.get_time() < end_time:
            pass