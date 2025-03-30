import time
import sys

try:
    import win32pdh
except ImportError:
    print("ERROR: win32pdh module not found.")
    print("Please install pywin32: pip install pywin32")
    sys.exit(1)

class PerformanceCounter:
    def __init__(self, counter_path):
        """Initialize a performance counter with the specified path."""
        self.counter_path = counter_path
        self.hQuery = win32pdh.OpenQuery()
        self.hCounter = win32pdh.AddCounter(self.hQuery, self.counter_path)
        
    def get_value(self):
        """Get the current value of the performance counter."""
        try:
            win32pdh.CollectQueryData(self.hQuery)
            type_flag, value = win32pdh.GetFormattedCounterValue(self.hCounter, win32pdh.PDH_FMT_DOUBLE)
            return value
        except Exception as e:
            print(f"Error getting counter value: {str(e)}")
            return None
            
    def close(self):
        """Close the performance counter."""
        try:
            win32pdh.RemoveCounter(self.hCounter)
            win32pdh.CloseQuery(self.hQuery)
        except Exception as e:
            print(f"Error closing counter: {str(e)}")

class CPUPerformanceCounters:
    """Helper class to manage CPU performance counters."""
    
    @staticmethod
    def get_core_usage(core_id):
        """Get the CPU usage percentage for a specific core."""
        counter_path = f"\\Processor({core_id})\\% Processor Time"
        counter = PerformanceCounter(counter_path)
        
        # First collection to initialize
        counter.get_value()
        
        # Wait a bit for accurate measurement
        time.sleep(0.1)
        
        # Get the actual value
        value = counter.get_value()
        counter.close()
        return value
        
    @staticmethod
    def get_all_core_usage():
        """Get CPU usage for all cores."""
        import psutil
        result = {}
        for i in range(psutil.cpu_count(logical=True)):
            result[i] = CPUPerformanceCounters.get_core_usage(i)
        return result
        
    @staticmethod
    def get_counter_paths():
        """Get available counter paths related to CPU performance."""
        try:
            # Refresh performance counter data
            win32pdh.EnumObjects(None, None, 0, 1)
            
            # Get processor-related counters
            processor_paths = []
            
            # Get processor object
            for object_name in win32pdh.EnumObjects(None, None, 0, 0):
                if object_name.lower() == 'processor':
                    # Get instances for processor
                    instances = win32pdh.EnumObjectItems(None, None, object_name, win32pdh.PERF_DETAIL_WIZARD, 0)[1]
                    
                    # Get counters for processor
                    counters = win32pdh.EnumObjectItems(None, None, object_name, win32pdh.PERF_DETAIL_WIZARD, 0)[0]
                    
                    # Create counter paths
                    for instance in instances:
                        for counter in counters:
                            path = f"\\{object_name}({instance})\\{counter}"
                            processor_paths.append(path)
                            
            return processor_paths
        except Exception as e:
            print(f"Error enumerating counter paths: {str(e)}")
            return []