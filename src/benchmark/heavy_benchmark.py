import time
import threading
import psutil
import win32pdh
import numpy as np
from src.benchmark.all_benchmark import AllBenchmark

class HeavyBenchmark(AllBenchmark):
    """Handles AVX benchmark tests (vector operations)."""
    
    def __init__(self, parent_benchmark):
        """Initialize with a reference to the parent benchmark."""
        super().__init__(parent_benchmark)
        
    def run_single_core_test(self, core_id, duration=10):
        """Run AVX test on a single core."""
        # Use base class to prepare test
        physical_core_id, progress_data, start_time = self.prepare_single_core_test(core_id, "heavy AVX load")
        
        # Prepare arrays for AVX operations - larger arrays for more stress
        array_size = 4000  # Increased array size for more intensive vector operations
        a = np.random.random(array_size).astype(np.float32)
        b = np.random.random(array_size).astype(np.float32)
        c = np.zeros(array_size, dtype=np.float32)
        d = np.random.random(array_size).astype(np.float32)
        e = np.random.random(array_size).astype(np.float32)
        
        # Prepare for a workload benchmark
        iterations = 0
        vector_ops = array_size * 15  # Each iteration now does multiple vector operations
        end_time = start_time + duration
        current_time = start_time
        
        # For periodic progress updates
        last_update_time = start_time
        update_interval = 0.4  # Update every 0.4 seconds
        
        # Execute AVX operations to stress the CPU
        while current_time < end_time and not self.parent._stop_event.is_set():
            # Chain of vector operations to fully utilize AVX units
            # Each iteration now does significantly more work
            for _ in range(5):  # Multiple passes of vector operations
                c = np.sin(a) * np.cos(b) + np.sqrt(np.abs(a * b))
                d = np.exp(np.abs(c) * 0.01) + np.log(np.abs(b) + 1.0)
                e = np.tanh(c) + np.power(d, 2) * np.sqrt(np.abs(a))
                a = d * e / (np.abs(c) + 0.01)
                b = np.arctan2(e, np.abs(d) + 0.01)
            
            iterations += 1
            
            # Update time
            current_time = self._get_precise_time()
            
            # Update progress based on time interval instead of iteration count
            if current_time - last_update_time >= update_interval:
                elapsed = current_time - start_time
                if elapsed > 0:
                    ops_per_sec = (iterations * vector_ops) / elapsed
                    # Use standardized logging
                    self.log_test_progress(core_id, "AVX", ops_per_sec, elapsed)
                    
                    # Store progress data point
                    self.collect_progress_data(progress_data, elapsed, ops_per_sec)
                    
                    # Update the last update time
                    last_update_time = current_time
        
        # Use base class to finalize results
        return self.finalize_single_core_result(
            core_id, start_time, iterations, vector_ops, 
            progress_data, "AVX", "vector ops"
        )
        
    def run_multithreaded_test(self, duration=10):
        """Run a multi-threaded AVX test using all available cores."""
        # Set up multithreaded test using base class
        thread_results, progress_data, stop_event, log_lock, progress_lock, overall_start = \
            self.setup_multithreaded_test("AVX")
        
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
                
            # Load AVX test implementation - increase intensity
            array_size = 4000  # Larger array size for more intensive computation
            a = np.random.random(array_size).astype(np.float32)
            b = np.random.random(array_size).astype(np.float32)
            c = np.zeros(array_size, dtype=np.float32)
            d = np.random.random(array_size).astype(np.float32)
            e = np.random.random(array_size).astype(np.float32)
            
            # Each iteration now does many more vector operations
            vector_ops = array_size * 15
            
            start_time = self._get_precise_time()
            iterations = 0
            thread_result = {'thread_id': thread_id, 'iterations': 0}

            # For periodic progress updates
            last_update_time = start_time
            update_interval = 0.4  # Update every 0.4 seconds
            
            while not stop_event.is_set():
                # Chain of vector operations to fully utilize AVX units
                for _ in range(5):  # Multiple passes of complex vector operations
                    c = np.sin(a) * np.cos(b) + np.sqrt(np.abs(a * b))
                    d = np.exp(np.abs(c) * 0.01) + np.log(np.abs(b) + 1.0)
                    e = np.tanh(c) + np.power(d, 2) * np.sqrt(np.abs(a))
                    a = d * e / (np.abs(c) + 0.01)
                    b = np.arctan2(e, np.abs(d) + 0.01)
                
                iterations += 1
                
                # Update progress based on time interval instead of iteration count
                current_time = self._get_precise_time()
                if current_time - last_update_time >= update_interval:
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        ops_per_sec = (iterations * vector_ops) / elapsed
                        
                        with log_lock:
                            # Use standardized logging
                            self.log_test_progress(thread_id, "AVX", ops_per_sec, elapsed, is_thread=True)
                        
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
            thread_result['vector_ops'] = vector_ops
            thread_result['total_ops'] = iterations * vector_ops
            thread_result['operations_per_second'] = (iterations * vector_ops) / elapsed
                
            # Record final data point
            with progress_lock:
                elapsed_since_start = end_time - overall_start
                progress_data.append(
                    self.create_thread_progress_data(thread_id, elapsed_since_start, 
                                                  (iterations * vector_ops) / elapsed)
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
            progress_data, progress_lock, log_lock, "AVX"
        )
        
        # Calculate overall statistics
        overall_end = self._get_precise_time()
        
        # Use base class to finalize results
        return self.finalize_multithreaded_result(
            thread_results, overall_start, overall_end, progress_data, "AVX", "vector ops"
        )