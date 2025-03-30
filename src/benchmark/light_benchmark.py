import time
import threading
import psutil
import win32pdh
import numpy as np
import sys
import ctypes

class LightBenchmark:
    """Handles light load benchmark tests using SSE2 integer operations."""
    
    def __init__(self, parent_benchmark):
        """Initialize with a reference to the parent benchmark."""
        self.parent = parent_benchmark
        # Load NumPy with optimizations enabled
        np.show_config()
        
    def run_single_core_test(self, core_id, duration=10):
        """Run light load integer test on a single core using SSE2."""
        physical_core_id = core_id // 2  # Calculate physical core ID
        
        # Use parent's logging and timing functions
        log = self.parent._log
        get_time = self.parent._get_precise_time
        
        # Collect initial performance data
        win32pdh.CollectQueryData(self.parent.perf_counters[core_id]['query'])
        
        # Prepare integer arrays for SSE2 operations
        # Using int32 which is ideal for SSE2 integer operations
        array_size = 256  # Small enough to fit in L1 cache
        a = np.ones(array_size, dtype=np.int32)
        b = np.ones(array_size, dtype=np.int32)
        c = np.zeros(array_size, dtype=np.int32)
        
        # Prepare for a light workload benchmark with integer operations
        iterations = 0
        ops_per_iteration = array_size * 4  # Each iteration processes array_size elements with 4 operations
        start_time = get_time()
        end_time = start_time + duration
        current_time = start_time
        
        # Initialize progress data collection
        progress_data = []
        
        # Execute simple integer operations that will use SSE2
        while current_time < end_time and not self.parent._stop_event.is_set():
            # Simple integer vector operations
            c = a + b       # Addition
            a = c * 2       # Multiplication
            b = c - a       # Subtraction
            c = a & b       # Bitwise AND
            
            iterations += 1
            
            # Update time and log progress periodically
            if iterations % 100000 == 0:
                current_time = get_time()
                elapsed = current_time - start_time
                if elapsed > 0:
                    ops_per_sec = (iterations * ops_per_iteration) / elapsed
                    log(f"Light int test, Core {physical_core_id}: {ops_per_sec:.2f} ops/sec (running for {elapsed:.2f}s)")
                    
                    # Store progress data point for graphing
                    progress_data.append({
                        'elapsed_seconds': elapsed,
                        'operations_per_second': ops_per_sec
                    })
                    
                    # Check time after measurement
                    current_time = get_time()
        
        # Collect final performance data
        win32pdh.CollectQueryData(self.parent.perf_counters[core_id]['query'])
        counter_value = win32pdh.GetFormattedCounterValue(
            self.parent.perf_counters[core_id]['counter'], 
            win32pdh.PDH_FMT_DOUBLE
        )
        
        # Calculate results
        elapsed = get_time() - start_time
        result = {}
        
        if elapsed > 0:
            ops_per_sec = (iterations * ops_per_iteration) / elapsed
            cpu_usage = counter_value[1]  # Extract the actual counter value
            
            # Add final data point
            if progress_data and elapsed > progress_data[-1]['elapsed_seconds']:
                progress_data.append({
                    'elapsed_seconds': elapsed,
                    'operations_per_second': ops_per_sec
                })
            
            result = {
                'iterations': iterations,
                'int_ops_per_iteration': ops_per_iteration,
                'total_ops': iterations * ops_per_iteration,
                'elapsed_seconds': elapsed,
                'operations_per_second': ops_per_sec,
                'cpu_usage_percent': cpu_usage,
                'progress': progress_data  # Store progress data in result
            }
            
            log(f"Light SSE2 int test on Core {physical_core_id} complete:")
            log(f"  Integer operations: {iterations * ops_per_iteration}")
            log(f"  Time: {elapsed:.2f} seconds")
            log(f"  Performance: {ops_per_sec:.2f} ops/sec")
            log(f"  CPU Usage: {cpu_usage:.2f}%")
            log(f"  Progress data points: {len(progress_data)}")
            
        return result
        
    def run_multithreaded_test(self, duration=10):
        """Run a multi-threaded light integer test using all available cores."""
        log = self.parent._log
        get_time = self.parent._get_precise_time
        cpu_count = self.parent.cpu_count
        
        log(f"Starting multi-threaded light integer SSE2 test with {cpu_count} threads...")
        
        # Create a shared stop event for all threads
        stop_event = threading.Event()
        
        # Create a list to hold results from each thread
        thread_results = []
        threads = []
        
        # Create locks for thread-safe operations
        log_lock = threading.Lock()
        progress_lock = threading.Lock()
        
        # Shared progress data
        overall_start = get_time()
        progress_data = []
        
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
                
            start_time = get_time()
            iterations = 0
            thread_result = {
                'thread_id': thread_id, 
                'iterations': 0,
                'int_ops_per_iteration': ops_per_iteration
            }
            
            # Light integer SSE2 load test implementation
            while not stop_event.is_set():
                # Simple integer vector operations using SSE2
                c = a + b       # Addition
                a = c * 2       # Multiplication
                b = c - a       # Subtraction
                c = a & b       # Bitwise AND
                
                iterations += 1
                
                # Every 100K iterations, update progress
                if iterations % 100000 == 0:
                    current_time = get_time()
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        ops_per_sec = (iterations * ops_per_iteration) / elapsed
                        
                        with log_lock:
                            log(f"MT light int test, Thread {thread_id}: {ops_per_sec:.2f} ops/sec")
                        
                        # Record progress data point with overall elapsed time
                        with progress_lock:
                            elapsed_since_start = current_time - overall_start
                            progress_data.append({
                                'elapsed_seconds': elapsed_since_start,
                                'thread_id': thread_id,
                                'operations_per_second': ops_per_sec
                            })
            
            # Record final statistics
            end_time = get_time()
            elapsed = end_time - start_time
            
            thread_result['iterations'] = iterations
            thread_result['elapsed_seconds'] = elapsed
            thread_result['total_ops'] = iterations * ops_per_iteration
            thread_result['operations_per_second'] = (iterations * ops_per_iteration) / elapsed
                
            # Record final data point
            with progress_lock:
                elapsed_since_start = end_time - overall_start
                progress_data.append({
                    'elapsed_seconds': elapsed_since_start,
                    'thread_id': thread_id,
                    'operations_per_second': (iterations * ops_per_iteration) / elapsed
                })
            
            with log_lock:
                thread_results.append(thread_result)
        
        # Start timer
        overall_start = get_time()
        
        # Start all threads
        for i in range(cpu_count):
            t = threading.Thread(target=thread_func, args=(i,))
            t.daemon = True
            threads.append(t)
            t.start()
        
        # Wait for duration
        time.sleep(duration)
        
        # Signal all threads to stop
        stop_event.set()
        
        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=2.0)  # Give threads 2 seconds to finish
        
        # Calculate overall statistics
        overall_end = get_time()
        overall_elapsed = overall_end - overall_start
        
        # Aggregate results
        total_iterations = sum(r['iterations'] for r in thread_results)
        total_ops = sum(r['total_ops'] for r in thread_results)
        avg_ops_per_sec = total_ops / overall_elapsed
        
        # Prepare result
        result = {
            'thread_count': len(thread_results),
            'total_iterations': total_iterations,
            'total_operations': total_ops,
            'elapsed_seconds': overall_elapsed,
            'operations_per_second': avg_ops_per_sec,
            'thread_results': thread_results,
            'progress': progress_data  # Store progress data in result
        }
        
        # Log results
        log(f"\nMulti-threaded light SSE2 integer test complete:")
        log(f"  Threads: {len(thread_results)}")
        log(f"  Total Operations: {total_ops:,}")
        log(f"  Time: {overall_elapsed:.2f} seconds")
        log(f"  Overall Performance: {avg_ops_per_sec:.2f} ops/sec")
        log(f"  Per Thread Average: {avg_ops_per_sec/len(thread_results):.2f} ops/sec")
        log(f"  Progress data points: {len(progress_data)}")
        
        return result
        
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
            self.parent._log(f"Warning: Could not check SSE2 support: {e}")
            # If we can't check, assume SSE2 is available (it's been standard since 2001)
            return True