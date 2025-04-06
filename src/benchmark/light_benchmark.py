import time
import threading
import psutil
import win32pdh
import numpy as np
import sys
import ctypes
from src.benchmark.all_benchmark import AllBenchmark

class LightBenchmark(AllBenchmark):
    """Handles SSE load benchmark tests using SSE2 integer operations."""
    
    def __init__(self, parent_benchmark):
        """Initialize with a reference to the parent benchmark."""
        super().__init__(parent_benchmark)
        # Load NumPy with optimizations enabled
        np.show_config()
        
    def run_single_core_test(self, core_id, duration=10):
        """Run test on a single core using SSE2."""
        # Use base class to prepare test
        physical_core_id, progress_data, start_time = self.prepare_single_core_test(core_id, "light load")
        
        # Prepare integer arrays for SSE2 operations
        # Using int32 which is ideal for SSE2 integer operations
        array_size = 256  # Small enough to fit in L1 cache
        a = np.ones(array_size, dtype=np.int32)
        b = np.ones(array_size, dtype=np.int32)
        c = np.zeros(array_size, dtype=np.int32)
        
        # Prepare for a workload benchmark with integer operations
        iterations = 0
        ops_per_iteration = array_size * 4  # Each iteration processes array_size elements with 4 operations
        end_time = start_time + duration
        current_time = start_time
        
        # For periodic progress updates
        last_update_time = start_time
        update_interval = 0.4  # Update every 0.4 seconds
        
        # Execute simple integer operations that will use SSE2
        while current_time < end_time and not self.parent._stop_event.is_set():
            # Simple integer vector operations
            c = a + b       # Addition
            a = c * 2       # Multiplication
            b = c - a       # Subtraction
            c = a & b       # Bitwise AND
            
            iterations += 1
            
            # Update time and check if it's time to log progress
            current_time = self._get_precise_time()
            if current_time - last_update_time >= update_interval:
                elapsed = current_time - start_time
                if elapsed > 0:
                    ops_per_sec = (iterations * ops_per_iteration) / elapsed
                    # Use standardized logging
                    self.log_test_progress(core_id, "SSE int", ops_per_sec, elapsed)
                    
                    # Store progress data point
                    self.collect_progress_data(progress_data, elapsed, ops_per_sec)
                    
                    # Update the last update time
                    last_update_time = current_time
                    
                    # Check time after measurement
                    current_time = self._get_precise_time()        

        # Use base class to finalize results
        return self.finalize_single_core_result(
            core_id, start_time, iterations, ops_per_iteration, 
            progress_data, "SSE2 int", "ops"
        )
        
    def run_multithreaded_test(self, duration=10):
        """Run a multi-threaded SSE test using all available cores."""
        # Set up multithreaded test using base class
        thread_results, progress_data, stop_event, log_lock, progress_lock, overall_start = \
            self.setup_multithreaded_test("SSE2 int")
        
        # Get CPU count from parent
        cpu_count = self.parent.cpu_count
        
        # Function that will run in each thread
        def thread_func(thread_id):
            # Set thread priority to highest available
            try:
                if self.parent.sys_platform == 'win32':
                    import win32api
                    import win32con
                    thread_handle = win32api.GetCurrentThread()
                    win32api.SetThreadPriority(thread_handle, win32con.THREAD_PRIORITY_TIME_CRITICAL)
            except Exception:
                pass  # Silently continue if we can't set thread priority
            
            # Prepare integer arrays for SSE2 operations
            array_size = 256  # Small enough to fit in L1 cache
            a = np.ones(array_size, dtype=np.int32)
            b = np.ones(array_size, dtype=np.int32)
            c = np.zeros(array_size, dtype=np.int32)
            
            ops_per_iteration = array_size * 4  # 4 operations per element per iteration
                
            start_time = self._get_precise_time()
            iterations = 0
            thread_result = {
                'thread_id': thread_id, 
                'iterations': 0,
                'ops_per_iteration': ops_per_iteration
            }
            
            # For periodic progress updates
            last_update_time = start_time
            update_interval = 0.4  # Update every 0.4 seconds

            # Integer SSE2 load test implementation
            while not stop_event.is_set():
                # Simple integer vector operations using SSE2
                c = a + b       # Addition
                a = c * 2       # Multiplication
                b = c - a       # Subtraction
                c = a & b       # Bitwise AND
                
                iterations += 1
                
                # Check if it's time to update progress
                current_time = self._get_precise_time()
                if current_time - last_update_time >= update_interval:
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        ops_per_sec = (iterations * ops_per_iteration) / elapsed
                        
                        with log_lock:
                            # Use standardized logging
                            self.log_test_progress(thread_id, "SSE", ops_per_sec, elapsed, is_thread=True)
                        
                        # Record progress data point with overall elapsed time
                        with progress_lock:
                            elapsed_since_start = current_time - overall_start
                            progress_data.append(
                                self.create_thread_progress_data(thread_id, elapsed_since_start, ops_per_sec)
                            )
                        
                        # Update the last update time
                        last_update_time = current_time
            
            # Record final statistics
            end_time = self._get_precise_time()
            elapsed = end_time - start_time
            
            thread_result['iterations'] = iterations
            thread_result['elapsed_seconds'] = elapsed
            thread_result['total_ops'] = iterations * ops_per_iteration
            thread_result['operations_per_second'] = (iterations * ops_per_iteration) / elapsed
                
            # Record final data point
            with progress_lock:
                elapsed_since_start = end_time - overall_start
                progress_data.append(
                    self.create_thread_progress_data(thread_id, elapsed_since_start, 
                                                  (iterations * ops_per_iteration) / elapsed)
                )
            
            with log_lock:
                thread_results.append(thread_result)
        
        # Start all threads
        threads = []
        for i in range(cpu_count):
            t = threading.Thread(target=thread_func, args=(i,))
            t.daemon = True
            threads.append(t)
            t.start()
        
        # Run threads for duration and monitor progress
        self.run_threads_for_duration(
            threads, duration, stop_event, overall_start, 
            progress_data, progress_lock, log_lock, "SSE int"
        )
        
        # Calculate overall statistics
        overall_end = self._get_precise_time()
        
        # Use base class to finalize results
        return self.finalize_multithreaded_result(
            thread_results, overall_start, overall_end, progress_data, "SSE2 int", "ops"
        )
        
    def _check_sse2_support(self):
        """Check if SSE2 is supported on the system."""
        # SSE2 is standard on all x86-64 CPUs
        if ctypes.sizeof(ctypes.c_voidp) == 8:  # 64-bit Python
            return True
            
        # For 32-bit systems, try to detect CPU features
        try:
            # Try platform-specific approaches
            if sys.platform == 'win32':
                # On Windows, we can use registry or WMI
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                    # Look for feature identifiers in the processor name
                    processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                    winreg.CloseKey(key)
                    
                    # Most modern CPUs have SSE2
                    return True
                except:
                    pass
                    
            # Fallback: try to use numpy's detection
            # NumPy will have been compiled with appropriate optimizations if available
            return hasattr(np, '__SSSE3__') or hasattr(np, '__SSE2__')
            
        except Exception as e:
            self._log(f"Warning: Could not check SSE2 support: {e}")
            # If we can't check, assume SSE2 is available (it's been standard since 2001)
            return True