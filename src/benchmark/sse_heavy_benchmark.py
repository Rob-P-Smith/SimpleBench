import time
import threading
import psutil
import random
import math
import ctypes
import numpy as np
from src.benchmark.all_benchmark import AllBenchmark

# Try to import SSE2 specific acceleration
try:
    import numba
    from numba import vectorize, float32, float64, int32, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class SSEHeavyBenchmark(AllBenchmark):
    """SSE-Heavy benchmark that focuses on SSE2 vector operations similar to Prime95.
    
    This benchmark performs Pi calculation using SSE2 instructions and focuses
    on operations that maximize CPU core and cache performance by using data sizes
    that fit within CPU cache (< 1024KB).
    """
    
    def __init__(self, parent_benchmark):
        """Initialize the SSE-Heavy benchmark.
        
        Args:
            parent_benchmark: The main benchmark controller that manages the benchmark process.
        """
        super().__init__(parent_benchmark)
        self.running = False
        self.stop_flag = False
        
        # Determine benchmark parameters for optimal cache usage (< 1024KB)
        # For SSE2, each vector is 128 bits (16 bytes)
        # For 4 float32 values per vector:
        self.vector_size = 16  # bytes
        self.max_vectors = 65536  # ~1024KB worth of vectors (1MB)
        self.iterations_per_sample = 1000  # Iterations before checking time
        
        # Setup Numba accelerated functions if available
        if NUMBA_AVAILABLE:
            self._setup_numba_functions()
            
    def _setup_numba_functions(self):
        """Setup Numba accelerated SSE2 functions."""
        # Define a Numba vectorized function that will use SSE2 on x86
        @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], target='parallel')
        def sse_vector_ops(a, b):
            # Series of operations that will use SSE2 instructions
            c = a * b
            d = c + a
            e = math.sin(d) + math.cos(c)
            f = e * e
            return f
        
        self.sse_vector_ops = sse_vector_ops
        
        # BBP Pi calculation algorithm (optimized for SSE)
        @numba.jit(nopython=True, parallel=True)
        def calculate_pi_term(k):
            return (4.0 / (8*k + 1)) - (2.0 / (8*k + 4)) - (1.0 / (8*k + 5)) - (1.0 / (8*k + 6))
        
        @numba.jit(nopython=True, parallel=True)
        def calculate_pi_bbp(n_terms):
            pi_sum = 0.0
            for k in range(n_terms):
                pi_sum += calculate_pi_term(k) / (16.0 ** k)
            return pi_sum
        
        self.calculate_pi_bbp = calculate_pi_bbp
    
    def _fallback_sse_simulation(self, arrays):
        """Fallback function that simulates SSE operations if Numba is not available.
        
        Still tries to utilize CPU and cache in a way similar to SSE operations.
        """
        a, b = arrays
        c = a * b
        d = c + a
        e = np.sin(d) + np.cos(c)
        f = e * e
        return f
    
    def _bbp_pi_calculation(self, iterations):
        """Bailey-Borwein-Plouffe algorithm for calculating Pi.
        
        This is a computationally intensive algorithm that's good for benchmarking.
        """
        if NUMBA_AVAILABLE:
            return self.calculate_pi_bbp(iterations)
        
        # Fallback implementation
        pi_sum = 0.0
        for k in range(iterations):
            term = (4.0 / (8*k + 1)) - (2.0 / (8*k + 4)) - (1.0 / (8*k + 5)) - (1.0 / (8*k + 6))
            pi_sum += term / (16.0 ** k)
        return pi_sum
        
    def _prepare_benchmark_data(self):
        """Prepare data arrays for SSE2 benchmark operations.
        
        Returns:
            tuple: (array_a, array_b) - NumPy arrays filled with test data
        """
        # Create arrays that fit within CPU cache (< 1024KB)
        # We'll create arrays of float32 (4 bytes each)
        # For SSE2, we'll process 4 float32 values at once
        n_elements = self.max_vectors * 4  # 4 float32 per SSE2 vector
        
        # Generate pseudorandom data for computation
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        array_a = np.random.random(n_elements).astype(np.float32)
        array_b = np.random.random(n_elements).astype(np.float32)
        
        return array_a, array_b
    
    def run_single_core_test(self, core_id, duration=10):
        """Run the SSE-Heavy benchmark on a single core.
        
        Args:
            core_id: The ID of the core to benchmark
            duration: Duration to run the benchmark in seconds
            
        Returns:
            dict: Benchmark results
        """
        # Use base class to prepare for test
        physical_core_id, progress_data, start_time = self.prepare_single_core_test(core_id, "SSE-Heavy load")
        
        # Prepare benchmark data
        arrays = self._prepare_benchmark_data()
        
        # Prepare for iterations
        iterations = 0
        vector_ops_per_iteration = 1000 * 4  # BBP with 1000 terms * operations per term
        end_time = start_time + duration
        current_time = start_time
        pi_values = []
        
        # For periodic progress updates
        last_update_time = start_time
        update_interval = 0.4  # Update every 0.4 seconds
        
        # Run the benchmark
        while current_time < end_time and not self.parent._stop_event.is_set():
            # First run: SSE vector operations on arrays
            if NUMBA_AVAILABLE:
                result = self.sse_vector_ops(arrays[0], arrays[1])
            else:
                result = self._fallback_sse_simulation(arrays)
            
            # Second run: Calculate Pi terms using BBP algorithm
            pi_approx = self._bbp_pi_calculation(1000)
            pi_values.append(pi_approx)
            
            iterations += 1
            
            # Update time and check if it's time to log progress
            current_time = self._get_precise_time()
            if current_time - last_update_time >= update_interval:
                elapsed = current_time - start_time
                if elapsed > 0:
                    ops_per_sec = (iterations * vector_ops_per_iteration) / elapsed
                    
                    # Use standardized logging
                    self.log_test_progress(core_id, "SSE int", ops_per_sec, elapsed)
                    
                    # Store progress data point
                    self.collect_progress_data(progress_data, elapsed, ops_per_sec)
                    
                    # Update the last update time
                    last_update_time = current_time
                    
                    # Check time after measurement
                    current_time = self._get_precise_time()
        
        # Calculate result stats
        result = self.finalize_single_core_result(
            core_id, start_time, iterations, vector_ops_per_iteration, 
            progress_data, "SSE", "vector ops"
        )
        
        # Add additional SSE-specific metrics
        result['pi_approximation'] = np.mean(pi_values) if pi_values else 0
        
        return result
    
    def run_multithreaded_test(self, duration=10):
        """Run the SSE-Heavy benchmark using all available cores.
        
        Args:
            duration: Duration to run the benchmark in seconds
            
        Returns:
            dict: Benchmark results
        """
        # Set up multithreaded test using base class
        thread_results, progress_data, stop_event, log_lock, progress_lock, overall_start = \
            self.setup_multithreaded_test("SSE")
        
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
            
            # Prepare benchmark data
            arrays = self._prepare_benchmark_data()
            
            # Each iteration involves intensive computations
            vector_ops_per_iteration = 1000 * 4  # BBP with 1000 terms * operations per term
            
            # Initialize counters
            start_time = self._get_precise_time()
            iterations = 0
            pi_values = []
            thread_result = {'thread_id': thread_id, 'iterations': 0}
            
            # For periodic progress updates
            last_update_time = start_time
            update_interval = 0.4  # Update every 0.4 seconds
            
            # Run the benchmark
            while not stop_event.is_set():
                # First run: SSE vector operations on arrays
                if NUMBA_AVAILABLE:
                    result = self.sse_vector_ops(arrays[0], arrays[1])
                else:
                    result = self._fallback_sse_simulation(arrays)
                
                # Second run: Calculate Pi terms using BBP algorithm
                pi_approx = self._bbp_pi_calculation(1000)
                pi_values.append(pi_approx)
                
                iterations += 1
                
                # Update progress at regular intervals
                current_time = self._get_precise_time()
                if current_time - last_update_time >= update_interval:
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        ops_per_sec = (iterations * vector_ops_per_iteration) / elapsed
                        
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
            thread_result['ops_per_iteration'] = vector_ops_per_iteration
            thread_result['total_ops'] = iterations * vector_ops_per_iteration
            thread_result['operations_per_second'] = (iterations * vector_ops_per_iteration) / elapsed
            thread_result['pi_approximation'] = np.mean(pi_values) if pi_values else 0
            
            # Record final data point
            with progress_lock:
                elapsed_since_start = end_time - overall_start
                progress_data.append(
                    self.create_thread_progress_data(thread_id, elapsed_since_start, 
                                                  (iterations * vector_ops_per_iteration) / elapsed)
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
            progress_data, progress_lock, log_lock, "SSE"
        )
        
        # Calculate overall statistics
        overall_end = self._get_precise_time()
        
        # Use base class to finalize results
        result = self.finalize_multithreaded_result(
            thread_results, overall_start, overall_end, progress_data, "SSE", "vector ops"
        )
        
        # Calculate and add Pi approximation average
        pi_values = [r.get('pi_approximation', 0) for r in thread_results]
        result['average_pi_approximation'] = sum(pi_values) / len(pi_values) if pi_values else 0
        
        return result
        
    def stop(self):
        """Stop the benchmark."""
        self.stop_flag = True
        return True