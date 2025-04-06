import time
import sys
import os

try:
    import win32pdh
    COUNTERS_AVAILABLE = True
except ImportError:
    print("WARNING: win32pdh module not found. Performance counters will be unavailable.")
    print("To enable performance monitoring: pip install pywin32")
    COUNTERS_AVAILABLE = False

class PerformanceCounter:
    def __init__(self, counter_path):
        """Initialize a performance counter with the specified path."""
        self.counter_path = counter_path
        self.initialized = False
        self.hQuery = None
        self.hCounter = None
        
        if not COUNTERS_AVAILABLE:
            return
            
        try:
            self.hQuery = win32pdh.OpenQuery()
            self.hCounter = win32pdh.AddCounter(self.hQuery, self.counter_path)
            self.initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize counter '{counter_path}': {str(e)}")
        
    def get_value(self):
        """Get the current value of the performance counter."""
        if not self.initialized:
            return 0.0
            
        try:
            # Call CollectQueryData twice with a longer delay between calls
            win32pdh.CollectQueryData(self.hQuery)
            time.sleep(0.5)  # Longer delay for stabilization
            win32pdh.CollectQueryData(self.hQuery)
            type_flag, value = win32pdh.GetFormattedCounterValue(self.hCounter, win32pdh.PDH_FMT_DOUBLE)
            return value
        except Exception as e:
            print(f"Error getting counter value: {str(e)}")
            return 0.0
            
    def close(self):
        """Close the performance counter."""
        if not self.initialized:
            return
            
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
        if not COUNTERS_AVAILABLE:
            return 0.0
            
        # Use _Total for overall CPU or specific core ID
        instance = "_Total" if core_id == -1 else str(core_id)
        counter_path = f"\\Processor({instance})\\% Processor Time"
        
        try:
            # Create a query that will persist during the collection
            query = win32pdh.OpenQuery()
            counter = win32pdh.AddCounter(query, counter_path)
            
            # First collection to initialize
            try:
                win32pdh.CollectQueryData(query)
            except:
                pass  # Ignore initial error
                
            # Wait for data to accumulate
            time.sleep(1.0)
            
            # Second collection
            win32pdh.CollectQueryData(query)
            type_flag, value = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
            
            # Clean up
            win32pdh.RemoveCounter(counter)
            win32pdh.CloseQuery(query)
            
            return value
        except Exception as e:
            print(f"Warning: Error collecting performance data: {str(e)}")
            return 0.0
        
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
        if not COUNTERS_AVAILABLE:
            return []
            
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
    
    @staticmethod
    def verify_counters_available():
        """Check if performance counters are available and working."""
        if not COUNTERS_AVAILABLE:
            print("Performance counters module (win32pdh) is not installed.")
            return False
            
        try:
            # Try to refresh counter list
            win32pdh.EnumObjects(None, None, 0, 1)
            
            # Try to access a simple counter
            test_counter = PerformanceCounter("\\System\\System Up Time")
            value = test_counter.get_value()
            test_counter.close()
            
            if value is not None and test_counter.initialized:
                print("Performance counters are available and working.")
                return True
            else:
                print("Performance counters are available but not returning values.")
                return False
        except Exception as e:
            print(f"Performance counters are not available: {str(e)}")
            return False
            
    @staticmethod
    def rebuild_performance_counters():
        """Attempt to rebuild performance counters (requires admin privileges)."""
        if not COUNTERS_AVAILABLE:
            return False
            
        print("Attempting to rebuild performance counters...")
        try:
            import subprocess
            # Run the lodctr /R command to rebuild performance counter registry
            result = subprocess.run(["lodctr", "/R"], capture_output=True, text=True)
            if result.returncode == 0:
                print("Successfully rebuilt performance counters.")
                return True
            else:
                print(f"Failed to rebuild performance counters: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error rebuilding performance counters: {str(e)}")
            return False
            
    @staticmethod
    def fix_counter_registry_permissions():
        """Try to fix performance counter registry permissions (requires admin)"""
        print("\nTo fix performance counter permissions, run the following commands as administrator:")
        print("1. Open Command Prompt as Administrator")
        print("2. Run: lodctr /R")
        print("3. Run: winmgmt /resyncperf")
        print("4. Restart your computer\n")
        
        try:
            # Check if we're running with admin rights
            is_admin = False
            try:
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            except:
                is_admin = False
                
            if is_admin:
                return CPUPerformanceCounters.rebuild_performance_counters()
            else:
                print("This script is not running with administrator privileges.")
                print("Please run the commands manually as described above.")
                return False
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
            
    @staticmethod
    def diagnose_counter_access():
        """Diagnose counter access issues."""
        if not COUNTERS_AVAILABLE:
            print("Performance counters module (win32pdh) is not installed.")
            return
            
        print("Starting counter access diagnosis...")
        
        # Try simpler counter paths first
        test_paths = [
            "\\System\\System Up Time",
            "\\Memory\\Available MBytes",
            "\\Processor(_Total)\\% Processor Time",
            "\\Processor(0)\\% Processor Time"
        ]
        
        for path in test_paths:
            print(f"\nTesting counter: {path}")
            counter = PerformanceCounter(path)
            
            if counter.initialized:
                print("  Counter initialized successfully")
                
                # First collection
                try:
                    win32pdh.CollectQueryData(counter.hQuery)
                    print("  First data collection successful")
                except Exception as e:
                    print(f"  First data collection failed: {e}")
                    
                # Wait
                time.sleep(0.5)
                
                # Second collection and get value
                try:
                    win32pdh.CollectQueryData(counter.hQuery)
                    type_flag, value = win32pdh.GetFormattedCounterValue(counter.hCounter, win32pdh.PDH_FMT_DOUBLE)
                    print(f"  Value retrieved: {value}")
                except Exception as e:
                    print(f"  Value retrieval failed: {e}")
                    
                counter.close()
            else:
                print("  Failed to initialize counter")
        
        # Check admin status
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            print(f"\nRunning as administrator: {is_admin}")
        except:
            print("\nCould not determine administrator status")
        
        print("\nDiagnosis complete.")