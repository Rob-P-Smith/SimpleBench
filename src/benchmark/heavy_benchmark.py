import time
import threading
import psutil
import win32pdh
import numpy as np

class HeavyBenchmark:
    """Handles heavy load AVX benchmark tests (vector operations)."""
    
    def __init__(self, parent_benchmark):
        """Initialize with a reference to the parent benchmark."""
        self.parent = parent_benchmark
        
    def run_single_core_test(self, core_id, duration=10):
        """Run heavy load AVX test on a single core."""
        physical_core_id = core_id // 2  # Calculate physical core ID
        
        # Use parent's logging and timing functions
        log = self.parent._log
        get_time = self.parent._get_precise_time
        
        # Collect initial performance data
        win32pdh.CollectQueryData(self.parent.perf_counters[core_id]['query'])
        
        # Prepare arrays for AVX operations - larger arrays for more stress
        array_size = 4000  # Increased array size for more intensive vector operations
        a = np.random.random(array_size).astype(np.float32)
        b = np.random.random(array_size).astype(np.float32)
        c = np.zeros(array_size, dtype=np.float32)
        d = np.random.random(array_size).astype(np.float32)
        e = np.random.random(array_size).astype(np.float32)
        
        # Initialize progress data collection
        progress_data = []
        
        # Prepare for a heavy workload benchmark
        iterations = 0
        vector_ops = array_size * 15  # Each iteration now does multiple vector operations
        start_time = get_time()
        end_time = start_time + duration
        current_time = start_time
        
        # Execute heavy AVX operations to stress the CPU
        while current_time < end_time and not self.parent._stop_event.is_set():
            # Chain of heavy vector operations to fully utilize AVX units
            # Each iteration now does significantly more work
            for _ in range(5):  # Multiple passes of vector operations
                c = np.sin(a) * np.cos(b) + np.sqrt(np.abs(a * b))
                d = np.exp(np.abs(c) * 0.01) + np.log(np.abs(b) + 1.0)
                e = np.tanh(c) + np.power(d, 2) * np.sqrt(np.abs(a))
                a = d * e / (np.abs(c) + 0.01)
                b = np.arctan2(e, np.abs(d) + 0.01)
            
            iterations += 1
            
            # Update progress every few iterations
            if iterations % 100 == 0:  # Reduced frequency due to heavier workload
                current_time = get_time()
                elapsed = current_time - start_time
                if elapsed > 0:
                    ops_per_sec = (iterations * vector_ops) / elapsed
                    log(f"Heavy AVX test, Core {physical_core_id}: {ops_per_sec:.2f} vector ops/sec (running for {elapsed:.2f}s)")
                    
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
            ops_per_sec = (iterations * vector_ops) / elapsed
            cpu_usage = counter_value[1]  # Extract the actual counter value
            
            # Add final data point if it's after the last one
            if progress_data and elapsed > progress_data[-1]['elapsed_seconds']:
                progress_data.append({
                    'elapsed_seconds': elapsed,
                    'operations_per_second': ops_per_sec
                })
            
            result = {
                'iterations': iterations,
                'vector_ops': vector_ops,
                'total_ops': iterations * vector_ops,
                'elapsed_seconds': elapsed,
                'operations_per_second': ops_per_sec,
                'cpu_usage_percent': cpu_usage,
                'progress': progress_data  # Add progress data to result
            }
            
            log(f"Heavy AVX load test on Core {physical_core_id} complete:")
            log(f"  Vector operations: {iterations * vector_ops}")
            log(f"  Time: {elapsed:.2f} seconds")
            log(f"  Performance: {ops_per_sec:.2f} vector ops/sec")
            log(f"  CPU Usage: {cpu_usage:.2f}%")
            log(f"  Progress data points: {len(progress_data)}")
            
        return result
        
    def run_multithreaded_test(self, duration=10):
        """Run a multi-threaded heavy AVX test using all available cores."""
        log = self.parent._log
        get_time = self.parent._get_precise_time
        cpu_count = self.parent.cpu_count
        
        log(f"Starting multi-threaded heavy AVX test with {cpu_count} threads...")
        
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
                
            # Heavy load AVX test implementation - increase intensity
            array_size = 4000  # Larger array size for more intensive computation
            a = np.random.random(array_size).astype(np.float32)
            b = np.random.random(array_size).astype(np.float32)
            c = np.zeros(array_size, dtype=np.float32)
            d = np.random.random(array_size).astype(np.float32)
            e = np.random.random(array_size).astype(np.float32)
            
            # Each iteration now does many more vector operations
            vector_ops = array_size * 15
            
            start_time = get_time()
            iterations = 0
            thread_result = {'thread_id': thread_id, 'iterations': 0}
            
            while not stop_event.is_set():
                # Chain of heavy vector operations to fully utilize AVX units
                for _ in range(5):  # Multiple passes of complex vector operations
                    c = np.sin(a) * np.cos(b) + np.sqrt(np.abs(a * b))
                    d = np.exp(np.abs(c) * 0.01) + np.log(np.abs(b) + 1.0)
                    e = np.tanh(c) + np.power(d, 2) * np.sqrt(np.abs(a))
                    a = d * e / (np.abs(c) + 0.01)
                    b = np.arctan2(e, np.abs(d) + 0.01)
                
                iterations += 1
                
                # Every few iterations, update progress
                if iterations % 100 == 0:  # Reduced frequency due to heavier workload
                    current_time = get_time()
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        ops_per_sec = (iterations * vector_ops) / elapsed
                        
                        with log_lock:
                            log(f"MT heavy test, Thread {thread_id}: {ops_per_sec:.2f} vector ops/sec")
                        
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
            thread_result['vector_ops'] = vector_ops
            thread_result['total_ops'] = iterations * vector_ops
            thread_result['operations_per_second'] = (iterations * vector_ops) / elapsed
                
            # Record final data point
            with progress_lock:
                elapsed_since_start = end_time - overall_start
                progress_data.append({
                    'elapsed_seconds': elapsed_since_start,
                    'thread_id': thread_id,
                    'operations_per_second': (iterations * vector_ops) / elapsed
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
            'elapsed_seconds': overall_elapsed,
            'operations_per_second': avg_ops_per_sec,
            'thread_results': thread_results,
            'progress': progress_data  # Add progress data to result
        }
        
        # Log results
        log(f"\nMulti-threaded heavy AVX test complete:")
        log(f"  Threads: {len(thread_results)}")
        log(f"  Total Vector Operations: {total_ops:,}")
        log(f"  Time: {overall_elapsed:.2f} seconds")
        log(f"  Overall Performance: {avg_ops_per_sec:,.2f} vector ops/sec")
        log(f"  Per Thread Average: {avg_ops_per_sec/len(thread_results):,.2f} vector ops/sec")
        log(f"  Progress data points: {len(progress_data)}")
        
        return result