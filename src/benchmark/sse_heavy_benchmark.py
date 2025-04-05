import time
import threading
import psutil
import random
import math
import ctypes
import numpy as np
from multiprocessing import cpu_count

# Try to import SSE2 specific acceleration
try:
    import numba
    from numba import vectorize, float32, float64, int32, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class SSEHeavyBenchmark:
    """SSE-Heavy benchmark that focuses on SSE2 vector operations similar to Prime95.
    
    This benchmark performs Pi calculation using SSE2 instructions and focuses
    on operations that maximize CPU core and cache performance by using data sizes
    that fit within CPU cache (< 1024KB).
    """
    
    def __init__(self, benchmark_controller):
        """Initialize the SSE-Heavy benchmark.
        
        Args:
            benchmark_controller: The main benchmark controller that manages the benchmark process.
        """
        self.controller = benchmark_controller
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
        
    def _worker_thread(self, thread_id, duration, result_dict, is_single_core=False):
        """Worker thread that runs the SSE2 benchmark operations.
        
        Args:
            thread_id: Identifier for this worker thread
            duration: Duration to run the benchmark in seconds
            result_dict: Dictionary to store results
            is_single_core: Whether this is a single core benchmark
        """
        # Prepare data that fits in cache
        arrays = self._prepare_benchmark_data()
        
        # Track iterations and timing
        start_time = self.controller._get_precise_time()
        end_time = start_time + duration
        iterations = 0
        pi_values = []
        
        # Run until the duration is reached
        while self.controller._get_precise_time() < end_time and not self.stop_flag:
            # First run: SSE vector operations on arrays
            if NUMBA_AVAILABLE:
                result = self.sse_vector_ops(arrays[0], arrays[1])
            else:
                result = self._fallback_sse_simulation(arrays)
            
            # Second run: Calculate Pi terms using BBP algorithm
            # Use a smaller number of terms to ensure it fits in cache
            pi_approx = self._bbp_pi_calculation(1000)
            pi_values.append(pi_approx)
            
            # Count as one full iteration
            iterations += 1
            
            # Every 10 iterations, update progress
            if iterations % 10 == 0 and is_single_core:
                elapsed = self.controller._get_precise_time() - start_time
                remaining = max(0, duration - elapsed)
                if self.controller.progress_callback:
                    # Calculate progress percentage
                    progress_percent = min(99, int((elapsed / duration) * 100))
                    self.controller.progress_callback(
                        f"SSE-Heavy Core {thread_id} benchmark: {progress_percent}% complete, "
                        f"{remaining:.1f}s remaining"
                    )
        
        # Record elapsed time and operations
        actual_time = self.controller._get_precise_time() - start_time
        
        # Store results
        result_dict[thread_id] = {
            'iterations': iterations,
            'time': actual_time,
            'operations_per_second': iterations / actual_time if actual_time > 0 else 0,
            'pi_approximation': np.mean(pi_values) if pi_values else 0
        }
    
    def run_single_core_test(self, core_id, duration):
        """Run the SSE-Heavy benchmark on a single core.
        
        Args:
            core_id: The ID of the core to benchmark
            duration: Duration to run the benchmark in seconds
            
        Returns:
            dict: Benchmark results
        """
        # Set up result storage
        result_dict = {}
        self.stop_flag = False
        
        # Log benchmark start
        physical_core_id = core_id // 2  # Convert logical to physical core ID
        self.controller._log(f"Starting SSE-Heavy benchmark on Core {physical_core_id} for {duration} seconds")
        
        try:
            # Run the worker thread
            self._worker_thread(core_id, duration, result_dict, is_single_core=True)
            
            # Extract results
            if core_id in result_dict:
                thread_result = result_dict[core_id]
                
                # Format results
                return {
                    'operations_per_second': thread_result['operations_per_second'],
                    'iterations': thread_result['iterations'],
                    'elapsed_seconds': thread_result['time'],
                    'pi_approximation': thread_result['pi_approximation']
                }
            else:
                return {
                    'operations_per_second': 0,
                    'iterations': 0,
                    'elapsed_seconds': duration,
                    'error': 'No results collected'
                }
                
        except Exception as e:
            self.controller._log(f"Error in SSE-Heavy benchmark on Core {physical_core_id}: {str(e)}")
            return {
                'operations_per_second': 0,
                'iterations': 0,
                'elapsed_seconds': duration,
                'error': str(e)
            }
    
    def run_multithreaded_test(self, duration):
        """Run the SSE-Heavy benchmark using all available cores.
        
        Args:
            duration: Duration to run the benchmark in seconds
            
        Returns:
            dict: Benchmark results
        """
        # Set up result storage and state
        result_dict = {}
        self.stop_flag = False
        
        # Get the number of threads to use (all logical processors)
        thread_count = psutil.cpu_count(logical=True)
        
        # Create and start worker threads
        threads = []
        start_time = self.controller._get_precise_time()
        
        self.controller._log(f"Starting SSE-Heavy multi-threaded benchmark using {thread_count} threads")
        
        try:
            # Start a worker thread for each logical core
            for i in range(thread_count):
                thread = threading.Thread(
                    target=self._worker_thread,
                    args=(i, duration, result_dict, False)
                )
                thread.daemon = True
                thread.start()
                threads.append(thread)
            
            # Wait for threads to finish
            for thread in threads:
                thread.join()
            
            # Calculate overall results
            total_operations = sum(result['iterations'] for result in result_dict.values())
            actual_duration = self.controller._get_precise_time() - start_time
            operations_per_second = total_operations / actual_duration if actual_duration > 0 else 0
            
            # Average Pi approximation across all threads
            pi_values = [result['pi_approximation'] for result in result_dict.values() if 'pi_approximation' in result]
            avg_pi = sum(pi_values) / len(pi_values) if pi_values else 0
            
            # Return the combined results
            return {
                'thread_count': thread_count,
                'thread_results': list(result_dict.values()),
                'total_operations': total_operations,
                'operations_per_second': operations_per_second,
                'elapsed_seconds': actual_duration,
                'average_pi_approximation': avg_pi
            }
            
        except Exception as e:
            self.controller._log(f"Error in SSE-Heavy multi-threaded benchmark: {str(e)}")
            return {
                'thread_count': thread_count,
                'operations_per_second': 0,
                'elapsed_seconds': duration,
                'error': str(e)
            }
    
    def stop(self):
        """Stop the benchmark."""
        self.stop_flag = True
        return True