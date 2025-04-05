import time
import threading
import psutil
import random
import math
import ctypes
import numpy as np
from multiprocessing import cpu_count

# Try to import AVX2 specific acceleration
try:
    import numba
    from numba import vectorize, float32, float64, int32, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class AVXHeavyBenchmark:
    """AVX-Heavy benchmark that focuses on AVX2 vector operations similar to Prime95.
    
    This benchmark performs Pi calculation using AVX2 instructions and focuses
    on operations that maximize CPU core and cache performance by using data sizes
    that fit within CPU cache (< 1024KB).
    """
    
    def __init__(self, benchmark_controller):
        """Initialize the AVX-Heavy benchmark.
        
        Args:
            benchmark_controller: The main benchmark controller that manages the benchmark process.
        """
        self.controller = benchmark_controller
        self.running = False
        self.stop_flag = False
        
        # Determine benchmark parameters for optimal cache usage (< 1024KB)
        # For AVX2, each vector is 256 bits (32 bytes)
        # For 8 float32 values per vector:
        self.vector_size = 32  # bytes
        self.max_vectors = 32768  # ~1024KB worth of vectors (1MB)
        self.iterations_per_sample = 1000  # Iterations before checking time
        
        # Setup Numba accelerated functions if available
        if NUMBA_AVAILABLE:
            self._setup_numba_functions()
            
    def _setup_numba_functions(self):
        """Setup Numba accelerated AVX2 functions."""
        # Define a Numba vectorized function that will use AVX2 on x86
        # target='parallel' lets Numba use SIMD instructions (AVX2 if available)
        @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], target='parallel')
        def avx_vector_ops(a, b):
            # Series of operations that will use AVX2 instructions when available
            # These operations are designed to be vectorizable
            c = a * b
            d = c * c + a
            e = math.sin(d) + math.cos(c)
            f = e * e + math.sqrt(abs(d))
            return f
        
        self.avx_vector_ops = avx_vector_ops
        
        # Bailey-Borwein-Plouffe Pi calculation algorithm (optimized for AVX)
        # This algorithm is more computationally intensive than the SSE version
        @numba.jit(nopython=True, parallel=True)
        def calculate_pi_term(k):
            return (4.0 / (8*k + 1)) - (2.0 / (8*k + 4)) - (1.0 / (8*k + 5)) - (1.0 / (8*k + 6))
        
        @numba.jit(nopython=True, parallel=True)
        def calculate_pi_bbp(n_terms):
            pi_sum = 0.0
            for k in range(n_terms):
                pi_sum += calculate_pi_term(k) / (16.0 ** k)
            return pi_sum
        
        # Chudnovsky algorithm - more compute-intensive than BBP
        @numba.jit(nopython=True, parallel=True)
        def factorial(n):
            if n == 0:
                return 1
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        
        @numba.jit(nopython=True, parallel=True)
        def chudnovsky_pi(iterations):
            sum = 0.0
            for k in range(iterations):
                numerator = factorial(6 * k) * (13591409 + 545140134 * k)
                denominator = factorial(3 * k) * (factorial(k) ** 3) * ((-640320) ** (3 * k))
                sum += numerator / denominator
            return 1 / (12 * sum) * math.sqrt(10005)
        
        self.calculate_pi_bbp = calculate_pi_bbp
        self.chudnovsky_pi = chudnovsky_pi
    
    def _fallback_avx_simulation(self, arrays):
        """Fallback function that simulates AVX operations if Numba is not available.
        
        Still tries to utilize CPU and cache in a way similar to AVX operations.
        """
        a, b = arrays
        c = a * b
        d = c * c + a
        e = np.sin(d) + np.cos(c)
        f = e * e + np.sqrt(np.abs(d))
        return f
    
    def _pi_calculation(self, iterations):
        """Calculate Pi using compute-intensive algorithms.
        
        This combines BBP and Chudnovsky algorithms that are good for benchmarking.
        """
        if NUMBA_AVAILABLE:
            # Use more iterations for BBP (lighter) and fewer for Chudnovsky (heavier)
            bbp_pi = self.calculate_pi_bbp(iterations)
            
            # Limit Chudnovsky iterations to avoid excessive compute time
            chud_iter = min(iterations // 20, 100)  
            try:
                chud_pi = self.chudnovsky_pi(chud_iter)
                return (bbp_pi + chud_pi) / 2
            except:
                # If Chudnovsky fails (can happen with large factorials), use BBP
                return bbp_pi
        
        # Fallback implementation (BBP only)
        pi_sum = 0.0
        for k in range(iterations):
            term = (4.0 / (8*k + 1)) - (2.0 / (8*k + 4)) - (1.0 / (8*k + 5)) - (1.0 / (8*k + 6))
            pi_sum += term / (16.0 ** k)
        return pi_sum
        
    def _prepare_benchmark_data(self):
        """Prepare data arrays for AVX2 benchmark operations.
        
        Returns:
            tuple: (array_a, array_b) - NumPy arrays filled with test data
        """
        # Create arrays that fit within CPU cache (< 1024KB)
        # We'll create arrays of float32 (4 bytes each)
        # For AVX2, we'll process 8 float32 values at once
        n_elements = self.max_vectors * 8  # 8 float32 per AVX2 vector
        
        # Generate pseudorandom data for computation
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        array_a = np.random.random(n_elements).astype(np.float32)
        array_b = np.random.random(n_elements).astype(np.float32)
        
        return array_a, array_b
        
    def _worker_thread(self, thread_id, duration, result_dict, is_single_core=False):
        """Worker thread that runs the AVX2 benchmark operations.
        
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
            # First run: AVX vector operations on arrays
            if NUMBA_AVAILABLE:
                result = self.avx_vector_ops(arrays[0], arrays[1])
            else:
                result = self._fallback_avx_simulation(arrays)
            
            # Second run: Calculate Pi terms using intensive algorithms
            # Use a smaller number of terms to ensure it fits in cache
            pi_approx = self._pi_calculation(1000)
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
                        f"AVX-Heavy Core {thread_id} benchmark: {progress_percent}% complete, "
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
        """Run the AVX-Heavy benchmark on a single core.
        
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
        self.controller._log(f"Starting AVX-Heavy benchmark on Core {physical_core_id} for {duration} seconds")
        
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
            self.controller._log(f"Error in AVX-Heavy benchmark on Core {physical_core_id}: {str(e)}")
            return {
                'operations_per_second': 0,
                'iterations': 0,
                'elapsed_seconds': duration,
                'error': str(e)
            }
    
    def run_multithreaded_test(self, duration):
        """Run the AVX-Heavy benchmark using all available cores.
        
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
        
        self.controller._log(f"Starting AVX-Heavy multi-threaded benchmark using {thread_count} threads")
        
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
            self.controller._log(f"Error in AVX-Heavy multi-threaded benchmark: {str(e)}")
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