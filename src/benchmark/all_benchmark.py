import time
import threading
import psutil
import numpy as np
import win32pdh
from datetime import datetime

class AllBenchmark:
    """Base class for all benchmark implementations with shared functionality."""
    
    def __init__(self, parent_benchmark):
        """Initialize with a reference to the parent benchmark controller.
        
        Args:
            parent_benchmark: Reference to the CPUBenchmark controller
        """
        self.parent = parent_benchmark
        self.stop_flag = False
        
    def _log(self, message):
        """Log a message using the parent's logging mechanism."""
        if hasattr(self.parent, '_log'):
            self.parent._log(message)
        else:
            print(message)
            
    def _get_precise_time(self):
        """Get precise time using the parent's timer."""
        if hasattr(self.parent, '_get_precise_time'):
            return self.parent._get_precise_time()
        else:
            return time.perf_counter()
    
    def log_test_progress(self, core_id, test_type, ops_per_sec, elapsed, is_thread=False):
        """Log standardized test progress message.
        
        Args:
            core_id: Core/thread ID being tested
            test_type: Type of test (SSE, AVX, etc.)
            ops_per_sec: Current operations per second
            elapsed: Elapsed time in seconds
            is_thread: Whether this is a thread in a multithreaded test
        """
        # Convert logical core ID to physical core ID for display
        physical_core_id = core_id // 2 if not isinstance(core_id, str) and not is_thread else core_id
        
        # Format the message with consistent pattern
        if is_thread:
            message = f"MT {test_type} test, Thread {core_id}: {ops_per_sec:.2f} ops/sec (running for {elapsed:.2f}s)"
        else:
            message = f"{test_type} test, Core {physical_core_id}: {ops_per_sec:.2f} ops/sec (running for {elapsed:.2f}s)"
            
        self._log(message)
        
    def log_test_completion(self, core_id, test_type, total_ops, elapsed, ops_per_sec, cpu_usage, progress_points, unit="ops", is_thread=False):
        """Log standardized test completion message.
        
        Args:
            core_id: Core/thread ID being tested
            test_type: Type of test (SSE, AVX, etc.)
            total_ops: Total operations completed
            elapsed: Elapsed time in seconds
            ops_per_sec: Operations per second
            cpu_usage: CPU usage percentage
            progress_points: Number of progress data points collected
            unit: Unit of measurement (ops, vector ops, etc.)
            is_thread: Whether this is a thread in a multithreaded test
        """
        # Convert logical core ID to physical core ID for display
        physical_core_id = core_id // 2 if not isinstance(core_id, str) and not is_thread else core_id
        
        # Format the completion message
        if is_thread:
            self._log(f"\nMT {test_type} test for Thread {core_id} complete:")
        else:
            self._log(f"{test_type} test on Core {physical_core_id} complete:")
            
        self._log(f"  {test_type.capitalize()} operations: {total_ops:,}")
        self._log(f"  Time: {elapsed:.2f} seconds")
        self._log(f"  Performance: {ops_per_sec:.2f} {unit}/sec")
        
        if cpu_usage is not None:
            self._log(f"  CPU Usage: {cpu_usage:.2f}%")
            
        self._log(f"  Progress data points: {progress_points}")
    
    def collect_progress_data(self, progress_data, elapsed, ops_per_sec):
        """Add a progress data point to the collection.
        
        Args:
            progress_data: List to add the data point to
            elapsed: Elapsed time in seconds
            ops_per_sec: Current operations per second
            
        Returns:
            Updated progress_data list
        """
        progress_data.append({
            'elapsed_seconds': elapsed,
            'operations_per_second': ops_per_sec
        })
        return progress_data
        
    def prepare_single_core_test(self, core_id, test_name):
        """Prepare for a single core benchmark test.
        
        Args:
            core_id: The logical core ID to test
            test_name: Name of the test for logging
            
        Returns:
            physical_core_id: The corresponding physical core ID
            progress_data: Empty list for collecting progress data
            start_time: Starting time of the test
        """
        physical_core_id = core_id // 2
        
        # Log test start
        self._log(f"Starting {test_name} benchmark on Core {physical_core_id}...")
        
        # Collect initial performance data if available
        if hasattr(self.parent, 'perf_counters') and core_id in self.parent.perf_counters:
            try:
                win32pdh.CollectQueryData(self.parent.perf_counters[core_id]['query'])
            except Exception:
                pass
        
        # Initialize progress data collection
        progress_data = []
        
        # Start timer
        start_time = self._get_precise_time()
        
        return physical_core_id, progress_data, start_time
        
    def collect_cpu_usage(self, core_id):
        """Collect CPU usage for a core from the performance counters.
        
        Args:
            core_id: The logical core ID to get CPU usage for
            
        Returns:
            CPU usage as a percentage or None if unavailable
        """
        if (hasattr(self.parent, 'perf_counters') and core_id in self.parent.perf_counters and 
                self.parent.perf_counters[core_id]['counter'] is not None):
            try:
                win32pdh.CollectQueryData(self.parent.perf_counters[core_id]['query'])
                counter_value = win32pdh.GetFormattedCounterValue(
                    self.parent.perf_counters[core_id]['counter'], 
                    win32pdh.PDH_FMT_DOUBLE
                )
                return counter_value[1]  # Extract the actual counter value
            except Exception:
                pass
        
        return None
        
    def finalize_single_core_result(self, core_id, start_time, iterations, ops_per_iteration, progress_data, test_type, unit="ops"):
        """Finalize results for a single core test.
        
        Args:
            core_id: The logical core ID that was tested
            start_time: Starting time of the test
            iterations: Number of iterations completed
            ops_per_iteration: Operations per iteration
            progress_data: Collected progress data points
            test_type: Type of test that was run
            unit: Unit of measurement (ops, vector ops, etc.)
            
        Returns:
            Dictionary with test results
        """
        physical_core_id = core_id // 2
        
        # Calculate elapsed time
        elapsed = self._get_precise_time() - start_time
        
        # Get CPU usage if available
        cpu_usage = self.collect_cpu_usage(core_id)
        
        # Calculate ops per second
        total_ops = iterations * ops_per_iteration
        ops_per_sec = total_ops / elapsed if elapsed > 0 else 0
        
        # Add final data point to progress data if needed
        if progress_data and elapsed > progress_data[-1]['elapsed_seconds']:
            self.collect_progress_data(progress_data, elapsed, ops_per_sec)
        
        # Log completion
        self.log_test_completion(
            core_id, test_type, total_ops, elapsed, ops_per_sec, 
            cpu_usage, len(progress_data), unit
        )
        
        # Return formatted result
        return {
            'iterations': iterations,
            'ops_per_iteration': ops_per_iteration,
            'total_ops': total_ops,
            'elapsed_seconds': elapsed,
            'operations_per_second': ops_per_sec,
            'cpu_usage_percent': cpu_usage,
            'progress': progress_data
        }
        
    def finalize_multithreaded_result(self, thread_results, overall_start, overall_end, progress_data, test_type, unit="ops"):
        """Finalize results for a multithreaded test.
        
        Args:
            thread_results: List of results from each thread
            overall_start: Starting time of the test
            overall_end: Ending time of the test
            progress_data: Collected progress data points
            test_type: Type of test that was run
            unit: Unit of measurement (ops, vector ops, etc.)
            
        Returns:
            Dictionary with test results
        """
        # Calculate overall elapsed time
        overall_elapsed = overall_end - overall_start
        
        # Aggregate results from all threads
        total_iterations = sum(r.get('iterations', 0) for r in thread_results)
        
        # Get total operations - handle different result formats
        total_ops = 0
        for r in thread_results:
            if 'total_ops' in r:
                total_ops += r['total_ops']
            elif 'iterations' in r and 'ops_per_iteration' in r:
                total_ops += r['iterations'] * r['ops_per_iteration']
            elif 'iterations' in r:
                # Estimate based on iterations if ops_per_iteration is missing
                total_ops += r['iterations']
                
        # Calculate overall operations per second
        avg_ops_per_sec = total_ops / overall_elapsed if overall_elapsed > 0 else 0
        
        # Log results
        self._log(f"\nMulti-threaded {test_type} test complete:")
        self._log(f"  Threads: {len(thread_results)}")
        self._log(f"  Total {test_type} operations: {total_ops:,}")
        self._log(f"  Time: {overall_elapsed:.2f} seconds")
        self._log(f"  Overall performance: {avg_ops_per_sec:,.2f} {unit}/sec")
        self._log(f"  Per thread average: {avg_ops_per_sec/len(thread_results):,.2f} {unit}/sec")
        self._log(f"  Progress data points: {len(progress_data)}")
        
        # Return formatted result
        return {
            'thread_count': len(thread_results),
            'total_iterations': total_iterations,
            'total_operations': total_ops,
            'elapsed_seconds': overall_elapsed,
            'operations_per_second': avg_ops_per_sec,
            'thread_results': thread_results,
            'progress': progress_data
        }
        
    def setup_multithreaded_test(self, test_name):
        """Set up a multithreaded benchmark test.
        
        Args:
            test_name: Name of the test for logging
            
        Returns:
            thread_results: Empty list for thread results
            progress_data: Empty list for progress data
            stop_event: Threading event for coordinating stop
            log_lock: Lock for thread-safe logging
            progress_lock: Lock for thread-safe progress updates
            start_time: Starting time of the test
        """
        # Log test start
        cpu_count = psutil.cpu_count(logical=True)
        self._log(f"Starting multi-threaded {test_name} test with {cpu_count} threads...")
        
        # Set up shared data structures
        thread_results = []
        progress_data = []
        
        # Create synchronization primitives
        stop_event = threading.Event()
        log_lock = threading.Lock()
        progress_lock = threading.Lock()
        
        # Start timer
        start_time = self._get_precise_time()
        
        return thread_results, progress_data, stop_event, log_lock, progress_lock, start_time
    
    def create_thread_progress_data(self, thread_id, elapsed_since_start, ops_per_sec):
        """Create a progress data point for a thread.
        
        Args:
            thread_id: ID of the thread
            elapsed_since_start: Overall elapsed time
            ops_per_sec: Operations per second
            
        Returns:
            Progress data point dictionary
        """
        return {
            'elapsed_seconds': elapsed_since_start,
            'thread_id': thread_id,
            'operations_per_second': ops_per_sec
        }
        
    def run_threads_for_duration(self, threads, duration, stop_event, start_time, 
                               progress_data, progress_lock, log_lock, test_name):
        """Run threads for a fixed duration with progress monitoring.
        
        Args:
            threads: List of thread objects
            duration: Duration to run in seconds
            stop_event: Event to signal threads to stop
            start_time: Starting time
            progress_data: List to store progress data
            progress_lock: Lock for thread-safe progress updates
            log_lock: Lock for thread-safe logging
            test_name: Name of the test for progress messages
            
        Returns:
            None
        """
        # For periodic progress updates
        last_update_time = start_time
        update_interval = 0.4  # Update every 0.4 seconds
        
        # Wait for duration
        time.sleep(duration)
        
        # Signal all threads to stop
        stop_event.set()
        
        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=2.0)  # Give threads 2 seconds to finish