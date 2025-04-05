import time
import threading
import psutil
import sys
import os
import contextlib
from datetime import datetime

# Use try/except for win32 modules
try:
    import win32pdh
    import win32process
    import win32api
    import win32con
except ImportError:
    print("ERROR: win32pdh or win32api modules not found.")
    print("Please install pywin32: pip install pywin32")
    sys.exit(1)

# Try to import cpuinfo, but it's optional
try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

from ctypes import byref, c_ulonglong
from src.benchmark.light_benchmark import LightBenchmark
from src.benchmark.heavy_benchmark import HeavyBenchmark
from src.benchmark.sse_heavy_benchmark import SSEHeavyBenchmark
from src.benchmark.avx_heavy_benchmark import AVXHeavyBenchmark
from src.utils.hpet_timer import HPETTimer
from src.utils.file_handler import create_log_file, save_results_to_file


class CPUBenchmark:
    def __init__(self):
        self.running = False
        self.results = {}
        self.current_core = None
        self.progress_callback = None
        self.completed_callback = None
        self._stop_event = threading.Event()
        self.sys_platform = sys.platform
        
        # Get CPU information
        self.cpu_count = psutil.cpu_count(logical=True)
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.threads_per_core = self.cpu_count // self.cpu_cores if self.cpu_cores > 0 else 1
        
        # For Ryzen systems, physical cores are odd-numbered logical processors
        self.physical_cores = self._identify_physical_cores()
        
        # Initialize HPET timer if available
        self.use_hpet = self._init_hpet()
        
        # Initialize performance counters
        self.perf_counters = {}
        self._init_perf_counters()
        
        # Initialize test modules
        self.light_benchmark = LightBenchmark(self)
        self.heavy_benchmark = HeavyBenchmark(self)
        self.sse_heavy_benchmark = SSEHeavyBenchmark(self)
        self.avx_heavy_benchmark = AVXHeavyBenchmark(self)
        
        # Initialize log file name
        self.log_filename = f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def _init_hpet(self):
        """Initialize High Precision Event Timer if available."""
        try:
            self.hpet_timer = HPETTimer()
            if self.hpet_timer.is_available():
                return True
            return False
        except Exception as e:
            print(f"Warning: Could not initialize HPET: {e}")
            return False
    
    # Remove the _initialize_log_file method and replace it with this:
    def _initialize_log_file(self):
        """Initialize the log file with header information."""
        try:
            # Create a new log file using the file_handler
            self.log_filename = create_log_file()
            
            # Open the log file for appending throughout the benchmark
            self.log_file = open(self.log_filename, "a")
            
            self._log(f"Log file initialized: {os.path.abspath(self.log_filename)}")
            return True
        except Exception as e:
            print(f"Error creating log file: {str(e)}")
            self.enable_logging = False
            return False

    def _log(self, message):
        """Log a message to the console and log file."""
        print(message)
        
        # Only write to log file if logging is enabled
        if hasattr(self, 'enable_logging') and self.enable_logging and hasattr(self, 'log_file'):
            try:
                self.log_file.write(message + '\n')
                self.log_file.flush()
            except Exception:
                pass  # Ignore errors writing to log file
                
        # Call progress callback if available
        if self.progress_callback:
            self.progress_callback(message)

    def _get_precise_time(self):
        """Get the current time using the most precise available timer."""
        if self.use_hpet:
            return self.hpet_timer.get_time()
        else:
            return time.perf_counter()
        
    def _init_perf_counters(self):
        """Initialize performance counters for each CPU core."""
        try:
            for i in range(self.cpu_count):
                counter_path = f"\\Processor({i})\\% Processor Time"
                self.perf_counters[i] = {}
                self.perf_counters[i]['query'] = win32pdh.OpenQuery()
                self.perf_counters[i]['counter'] = win32pdh.AddCounter(
                    self.perf_counters[i]['query'], 
                    counter_path
                )
                
            self._log(f"Initialized performance counters for {self.cpu_count} logical processors")
        except Exception as e:
            self._log(f"Error initializing performance counters: {str(e)}")

    def _identify_physical_cores(self):
        """Identify physical cores in the system."""
        try:
            # First attempt: try to get physical core mapping from psutil
            physical_cores = {}
            logical_cores = psutil.cpu_count(logical=True)
            
            # Different approach based on CPU vendor
            cpu_info = {}
            if CPUINFO_AVAILABLE:
                cpu_info = cpuinfo.get_cpu_info()
                
            else:
                # Fallback for basic CPU info
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                      r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                    processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                    vendor_id = winreg.QueryValueEx(key, "VendorIdentifier")[0]
                    winreg.CloseKey(key)
                    
                    cpu_info = {'brand': processor_name, 'vendor_id': vendor_id}
                except Exception:
                    cpu_info = {'brand': 'Unknown', 'vendor_id': 'Unknown'}
            
            # For AMD Ryzen, typically odd logical processors are physical cores
            if 'vendor_id' in cpu_info and 'amd' in cpu_info['vendor_id'].lower():
                # AMD specific core identification
                for i in range(logical_cores):
                    physical_cores[i] = (i % 2 == 0)  # Even cores are physical
                    
            else:
                # Default for Intel and unknown CPUs - assume all are physical for benchmarking
                for i in range(logical_cores):
                    physical_cores[i] = True  # Treat all as physical
                    
            
            return physical_cores
        except Exception as e:
            print(f"Error identifying physical cores: {str(e)}")
            # If we can't identify, assume all cores are physical for benchmarking
            return {i: True for i in range(psutil.cpu_count(logical=True))}

    def _set_realtime_priority(self):
        """Set process to real-time priority."""
        try:
            if sys.platform == 'win32':
                self.original_priority = win32process.GetPriorityClass(win32process.GetCurrentProcess())
                win32process.SetPriorityClass(win32process.GetCurrentProcess(), win32con.REALTIME_PRIORITY_CLASS)
                self._log("Set process priority to real-time for benchmark")
                return True
            else:
                # For non-Windows platforms, try to use os.nice
                self.original_priority = os.nice(0)
                os.nice(-20)  # Lowest niceness = highest priority
                self._log("Set process priority to maximum for benchmark")
                return True
        except Exception as e:
            self._log(f"Warning: Could not set high priority: {str(e)}")
            return False

    def _restore_priority(self):
        """Restore original process priority."""
        try:
            if hasattr(self, 'original_priority') and self.original_priority is not None:
                if sys.platform == 'win32':
                    win32process.SetPriorityClass(win32process.GetCurrentProcess(), self.original_priority)
                else:
                    current = os.nice(0)
                    os.nice(self.original_priority - current)  # Adjust back to original
                self._log("Restored original process priority")
                self.original_priority = None
                return True
        except Exception as e:
            self._log(f"Warning: Could not restore original priority: {str(e)}")
            return False

    @contextlib.contextmanager
    def high_priority(self):
        """Context manager for running code with high priority."""
        raised_priority = self._set_realtime_priority()
        try:
            yield
        finally:
            if raised_priority:
                self._restore_priority()
                
    def _log_test_start(self, core_id, test_type):
        """Log test start time to a file, blocking execution while writing."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            core_desc = "multithreaded" if core_id == "multithreaded" else f"Core {core_id//2}"
            
            with open(self.log_filename, "a") as f:
                f.write(f"{timestamp} - Starting {test_type} test on {core_desc}\n")
                f.flush()  # Ensure data is written to disk
                
            return True
        except Exception as e:
            self._log(f"Error writing to log file: {str(e)}")
            return False

    def _benchmark_core(self, core_id, duration=10):
        """Run benchmark on specific core for specified duration."""
        if not self.running:
            return
            
        self.current_core = core_id
        physical_core_id = core_id // 2  # Calculate physical core ID
        self.results[core_id] = {}
        
        # Set process affinity to this core
        process = psutil.Process()
        original_affinity = process.cpu_affinity()
        
        try:
            # Set affinity to only use the specified core
            process.cpu_affinity([core_id])
            
            # Run light load test first
            self._log(f"Starting light load benchmark on Core {physical_core_id}...")
            light_result = self.light_benchmark.run_single_core_test(core_id, duration)
            self.results[core_id]['light'] = light_result
            
            # Check if we should continue
            if self._stop_event.is_set():
                return
                
            # Run heavy load test next
            self._log(f"Starting heavy AVX load benchmark on Core {physical_core_id}...")
            heavy_result = self.heavy_benchmark.run_single_core_test(core_id, duration)
            self.results[core_id]['heavy'] = heavy_result
                
        except Exception as e:
            self._log(f"Error during benchmark on Core {physical_core_id}: {str(e)}")
        finally:
            # Restore original affinity
            try:
                process.cpu_affinity(original_affinity)
            except Exception:
                pass

    def start(self, progress_callback=None, completed_callback=None, duration_per_core=10,
             run_light_tests=True, run_heavy_tests=False, run_sse_heavy_tests=False, 
             run_avx_heavy_tests=False, run_multicore_tests=True, enable_logging=True):
        """Start the benchmark process."""
        if self.running:
            return False
            
        self.running = True
        self._stop_event.clear()
        self.progress_callback = progress_callback
        self.completed_callback = completed_callback
        self.results = {}
        
        # Initialize log file if logging is enabled
        self.enable_logging = enable_logging
        if enable_logging:
            self._initialize_log_file()
        else:
            # Set up a dummy log file attribute to avoid errors
            self.log_file = None
        
        # Start benchmark in a separate thread
        self._benchmark_thread = threading.Thread(
            target=self._run_benchmark,
            args=(duration_per_core, run_light_tests, run_heavy_tests, 
                  run_sse_heavy_tests, run_avx_heavy_tests, run_multicore_tests)
        )
        self._benchmark_thread.daemon = True
        self._benchmark_thread.start()
        
        return True
    
    def _run_benchmark(self, duration_per_core, run_light_tests, run_heavy_tests, 
                       run_sse_heavy_tests, run_avx_heavy_tests, run_multicore_tests):
        """Run the benchmark suite with the specified options."""
        try:
            # Get physical cores to test
            core_ids = self._get_test_core_ids()
            
            # Run light load tests on each physical core
            if run_light_tests:
                self._log("Starting light load tests on individual cores")
                self._run_single_core_tests(core_ids, 'light', duration_per_core)
            
            # Run heavy load tests on each physical core
            if run_heavy_tests:
                self._log("Starting heavy load tests on individual cores")
                self._run_single_core_tests(core_ids, 'heavy', duration_per_core)
            
            # Run SSE-Heavy load tests on each physical core
            if run_sse_heavy_tests:
                self._log("Starting SSE-Heavy load tests on individual cores")
                self._run_single_core_tests(core_ids, 'sse-heavy', duration_per_core)
            
            # Run AVX-Heavy load tests on each physical core
            if run_avx_heavy_tests:
                self._log("Starting AVX-Heavy load tests on individual cores")
                self._run_single_core_tests(core_ids, 'avx-heavy', duration_per_core)
            
            # Run multi-threaded tests
            if run_multicore_tests:
                self._log("\nStarting multi-threaded tests using all cores")
                self._run_multicore_tests(duration_per_core, run_light_tests, run_heavy_tests, 
                                        run_sse_heavy_tests, run_avx_heavy_tests)
            
            # Complete the benchmark
            self._log("\nAll benchmarks completed!")
            
            # Summarize results
            self._summarize_results()
        
        except Exception as e:
            self._log(f"Error during benchmark: {str(e)}")
        finally:
            # Close log file if it was opened
            if self.enable_logging and hasattr(self, 'log_file') and self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass
                    
            self.running = False
            if self.completed_callback:
                self.completed_callback()

    def _get_test_core_ids(self):
        """Get the list of cores to test - only one logical core per physical core."""
        physical_cores_seen = set()
        cores_to_test = []
        
        for core_id in sorted(self.physical_cores.keys()):
            if self.physical_cores[core_id]:
                physical_core_id = core_id // 2
                if physical_core_id not in physical_cores_seen:
                    cores_to_test.append(core_id)
                    physical_cores_seen.add(physical_core_id)
                    
        return cores_to_test
                
    def _run_single_core_tests(self, core_ids, test_type, duration):
        """Run tests of the specified type on each core."""
        for core_id in core_ids:
            # Check if we should stop
            if self._stop_event.is_set():
                return
                
            # Get only the physical cores
            if core_id in self.physical_cores:
                physical_core_id = core_id // 2  # Calculate physical core ID
                
                # Log test start time to file (blocking operation)
                self._log_test_start(core_id, test_type)
                
                # Set process affinity to this core
                process = psutil.Process()
                original_affinity = process.cpu_affinity()
                
                try:
                    # Set affinity to only use the specified core
                    process.cpu_affinity([core_id])
                    
                    # Initialize core results if needed
                    if core_id not in self.results:
                        self.results[core_id] = {}
                    
                    if test_type == 'light':
                        # Light load test
                        self._log(f"Starting light load benchmark on Core {physical_core_id}...")
                        self.current_core = core_id
                        light_result = self.light_benchmark.run_single_core_test(core_id, duration)
                        self.results[core_id]['light'] = light_result
                        
                    elif test_type == 'heavy':
                        # Heavy load test
                        self._log(f"Starting heavy AVX load benchmark on Core {physical_core_id}...")
                        self.current_core = core_id
                        heavy_result = self.heavy_benchmark.run_single_core_test(core_id, duration)
                        self.results[core_id]['heavy'] = heavy_result
                        
                    elif test_type == 'sse-heavy':
                        # SSE-Heavy load test
                        self._log(f"Starting SSE-Heavy load benchmark on Core {physical_core_id}...")
                        self.current_core = core_id
                        sse_heavy_result = self.sse_heavy_benchmark.run_single_core_test(core_id, duration)
                        self.results[core_id]['sse-heavy'] = sse_heavy_result
                        
                    elif test_type == 'avx-heavy':
                        # AVX-Heavy load test
                        self._log(f"Starting AVX-Heavy load benchmark on Core {physical_core_id}...")
                        self.current_core = core_id
                        avx_heavy_result = self.avx_heavy_benchmark.run_single_core_test(core_id, duration)
                        self.results[core_id]['avx-heavy'] = avx_heavy_result
                        
                finally:
                    # Restore original affinity
                    try:
                        process.cpu_affinity(original_affinity)
                    except Exception:
                        pass

    def _run_multicore_tests(self, duration, run_light_tests, run_heavy_tests, 
                           run_sse_heavy_tests, run_avx_heavy_tests):
        """Run multi-threaded benchmarks."""
        # Initialize multicore results container
        self.results['multithreaded'] = {}
        
        try:
            # Run light load multithreaded test
            if run_light_tests:
                # Log test start time to file (blocking operation)
                self._log_test_start('multithreaded', 'light')
                
                self._log("Running multi-threaded light load test...")
                self.current_core = 'multithreaded-light'
                light_result = self.light_benchmark.run_multithreaded_test(duration)
                self.results['multithreaded']['light'] = light_result
                
                self._log(f"Light load multi-threaded test complete.")
                self._log(f"  Total operations: {light_result['total_operations']:,}")
                self._log(f"  Time: {light_result['elapsed_seconds']:.2f} seconds")
                self._log(f"  Overall performance: {light_result['operations_per_second']:,.2f} ops/sec")
            
            # Run heavy load multithreaded test
            if run_heavy_tests:
                # Log test start time to file (blocking operation)
                self._log_test_start('multithreaded', 'heavy')
                
                self._log("Running multi-threaded heavy load AVX test...")
                self.current_core = 'multithreaded-heavy'
                heavy_result = self.heavy_benchmark.run_multithreaded_test(duration)
                self.results['multithreaded']['heavy'] = heavy_result
                
                self._log(f"Heavy load multi-threaded test complete.")
                self._log(f"  Time: {heavy_result['elapsed_seconds']:.2f} seconds")
                self._log(f"  Overall performance: {heavy_result['operations_per_second']:,.2f} ops/sec")
                
            # Run SSE-Heavy load multithreaded test
            if run_sse_heavy_tests:
                # Log test start time to file (blocking operation)
                self._log_test_start('multithreaded', 'sse-heavy')
                
                self._log("Running multi-threaded SSE-Heavy load test...")
                self.current_core = 'multithreaded-sse-heavy'
                sse_heavy_result = self.sse_heavy_benchmark.run_multithreaded_test(duration)
                self.results['multithreaded']['sse-heavy'] = sse_heavy_result
                
                self._log(f"SSE-Heavy load multi-threaded test complete.")
                self._log(f"  Time: {sse_heavy_result['elapsed_seconds']:.2f} seconds")
                self._log(f"  Overall performance: {sse_heavy_result['operations_per_second']:,.2f} ops/sec")
                
            # Run AVX-Heavy load multithreaded test
            if run_avx_heavy_tests:
                # Log test start time to file (blocking operation)
                self._log_test_start('multithreaded', 'avx-heavy')
                
                self._log("Running multi-threaded AVX-Heavy load test...")
                self.current_core = 'multithreaded-avx-heavy'
                avx_heavy_result = self.avx_heavy_benchmark.run_multithreaded_test(duration)
                self.results['multithreaded']['avx-heavy'] = avx_heavy_result
                
                self._log(f"AVX-Heavy load multi-threaded test complete.")
                self._log(f"  Time: {avx_heavy_result['elapsed_seconds']:.2f} seconds")
                self._log(f"  Overall performance: {avx_heavy_result['operations_per_second']:,.2f} ops/sec")
            
        except Exception as e:
            self._log(f"Error in multi-threaded tests: {str(e)}")
        finally:
            self.current_core = None

    def stop(self):
        """Stop the benchmark process."""
        self._stop_event.set()
        self._log("Benchmark stopping...")
        
        # Close log file if it was opened
        if hasattr(self, 'enable_logging') and self.enable_logging and hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
                
        return True

    def _summarize_results(self):
        """Summarize benchmark results."""
        if not self.results:
            self._log("No benchmark results to summarize")
            return
            
        try:
            # Test types we want to collect results for
            test_types = ['light', 'heavy', 'sse-heavy', 'avx-heavy']
            test_results = {}
            
            # Collect results for each test type
            for test_type in test_types:
                test_results[test_type] = {}
                for core_id, data in self.results.items():
                    if core_id == 'multithreaded':
                        continue  # Skip multithreaded results for now
                    
                    if (test_type in data and isinstance(data[test_type], dict) 
                            and 'operations_per_second' in data[test_type]):
                        test_results[test_type][core_id] = data[test_type]
            
            # Calculate maximum width for formatting
            max_width = max(
                60,  # Minimum width
                len("CPU BENCHMARK RESULTS SUMMARY"),
                len("LIGHT LOAD TEST RESULTS (Simple Operations)"),
                len("HEAVY LOAD TEST RESULTS (AVX Vector Operations)"),
                len("SSE-HEAVY LOAD TEST RESULTS (SSE Vector Operations)"),
                len("AVX-HEAVY LOAD TEST RESULTS (Advanced AVX Vector Operations)"),
                len("MULTI-THREADED TEST RESULTS")
            )
            
            # Create the box with equals and pipes
            border = "=" * (max_width + 4)
            self._log("\n" + border)
            
            # Header
            title = "CPU BENCHMARK RESULTS SUMMARY"
            self._log(f"| {title.center(max_width)} |")
            
            # Summarize results for each test type with descriptive headers
            test_headers = {
                'light': "LIGHT LOAD TEST RESULTS (Simple Operations)",
                'heavy': "HEAVY LOAD TEST RESULTS (AVX Vector Operations)",
                'sse-heavy': "SSE-HEAVY LOAD TEST RESULTS (SSE Vector Operations)",
                'avx-heavy': "AVX-HEAVY LOAD TEST RESULTS (Advanced AVX Vector Operations)"
            }
            
            # Display results for each test type
            for test_type, header in test_headers.items():
                if test_results[test_type]:
                    self._summarize_test_results(header, test_results[test_type], max_width, test_type)
            
            # Summarize multi-threaded test results if available
            if 'multithreaded' in self.results:
                self._summarize_multithreaded_results(max_width)
            
            # Close the box
            self._log(border)
            
            # Add log file location
            if hasattr(self, 'enable_logging') and self.enable_logging:
                self._log(f"\nBenchmark log file: {os.path.abspath(self.log_filename)}")
            
            # Save parsed results to results directory if logging is enabled
            if hasattr(self, 'enable_logging') and self.enable_logging:
                try:
                    # Format results for saving as a clean summary
                    formatted_results = []
                    formatted_results.append(f"CPU BENCHMARK RESULTS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    formatted_results.append("=" * 80)
                    formatted_results.append("")
                    
                    # System information 
                    formatted_results.append("SYSTEM INFORMATION")
                    formatted_results.append("-" * 20)
                    formatted_results.append(f"CPU cores: {self.cpu_cores} physical, {self.cpu_count} logical")
                    formatted_results.append("")
                    
                    # Add summary for each test type
                    for test_type, header in test_headers.items():
                        if test_results[test_type]:
                            formatted_results.append(header)
                            formatted_results.append("-" * len(header))
                            
                            # Calculate statistics
                            ops_per_sec_values = [r['operations_per_second'] for r in test_results[test_type].values()]
                            avg_ops = sum(ops_per_sec_values) / len(test_results[test_type])
                            max_ops = max(ops_per_sec_values)
                            min_ops = min(ops_per_sec_values)
                            
                            # Determine appropriate unit based on test type
                            unit = "ops/sec"
                            if test_type in ['heavy', 'sse-heavy', 'avx-heavy']:
                                unit = "vector ops/sec"
                            
                            # Add average performance
                            formatted_results.append(f"Average {unit} per core: {avg_ops:.2f}")
                            formatted_results.append(f"Maximum {unit}: {max_ops:.2f}")
                            formatted_results.append(f"Minimum {unit}: {min_ops:.2f}")
                            formatted_results.append("")
                            
                            # Add core ranking
                            formatted_results.append("CORE PERFORMANCE RANKING (Higher is Better)")
                            
                            # Rank cores by performance
                            ranked_cores = sorted(
                                [(core_id, data['operations_per_second']) for core_id, data in test_results[test_type].items()],
                                key=lambda x: x[1], 
                                reverse=True
                            )
                            
                            # List cores ranked by performance
                            for i, (logical_core_id, ops) in enumerate(ranked_cores):
                                rank = i + 1
                                physical_core_id = logical_core_id // 2
                                deviation_pct = (ops/avg_ops - 1) * 100
                                formatted_results.append(f"Core {physical_core_id}: {ops:.2f} {unit}  #{rank}  ({deviation_pct:+.2f}% from mean)")
                            
                            formatted_results.append("")
                    
                    # Add multi-threaded results if available
                    if 'multithreaded' in self.results:
                        formatted_results.append("MULTI-THREADED TEST RESULTS")
                        formatted_results.append("-" * 30)
                        
                        for test_type, test_header, ops_unit in [
                            ('light', 'Light Load Multi-Threaded Test:', 'ops/sec'),
                            ('heavy', 'Heavy Load (AVX) Multi-Threaded Test:', 'vector ops/sec'),
                            ('sse-heavy', 'SSE-Heavy Load Multi-Threaded Test:', 'vector ops/sec'),
                            ('avx-heavy', 'AVX-Heavy Load Multi-Threaded Test:', 'vector ops/sec')
                        ]:
                            if test_type in self.results['multithreaded']:
                                result = self.results['multithreaded'][test_type]
                                formatted_results.append(f"{test_header}")
                                formatted_results.append(f"  Threads: {result['thread_count']}")
                                formatted_results.append(f"  Overall performance: {result['operations_per_second']:,.2f} {ops_unit}")
                                
                                # Calculate scaling efficiency if we have single-core results
                                if any(k != 'multithreaded' for k in self.results.keys()):
                                    single_core_results = [data[test_type]['operations_per_second'] 
                                                        for core_id, data in self.results.items() 
                                                        if core_id != 'multithreaded' and test_type in data]
                                    if single_core_results:
                                        single_core_avg = sum(single_core_results) / len(single_core_results)
                                        scaling_ratio = result['operations_per_second'] / (single_core_avg * result['thread_count'])
                                        formatted_results.append(f"  Scaling efficiency: {scaling_ratio*100:.1f}%")
                                
                                formatted_results.append("")
                    
                    # Save the formatted results to results directory
                    from src.utils.file_handler import save_results_to_file
                    results_filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    results_path = save_results_to_file(formatted_results, results_filename)
                    self._log(f"Benchmark results summary saved to: {os.path.abspath(results_path)}")
                    
                except Exception as e:
                    self._log(f"Error saving results to file: {str(e)}")
        
        except Exception as e:
            self._log(f"Error summarizing results: {str(e)}")
    def _summarize_test_results(self, header, results, max_width, test_type='light'):
        """Summarize results for a specific test type."""
        # Divider line
        self._log("|" + "-" * (max_width + 2) + "|")
        
        # Header for this test
        self._log(f"| {header.center(max_width)} |")
        
        # Divider line
        self._log("|" + "-" * (max_width + 2) + "|")
        
        # Calculate statistics
        ops_per_sec_values = [r['operations_per_second'] for r in results.values()]
        avg_ops = sum(ops_per_sec_values) / len(results)
        max_ops = max(ops_per_sec_values)
        min_ops = min(ops_per_sec_values)
        
        # Determine appropriate unit based on test type
        unit = "ops/sec"
        if test_type in ['heavy', 'sse-heavy', 'avx-heavy']:
            unit = "vector ops/sec"
        
        # Average performance
        self._log(f"| Average {unit} per core: {avg_ops:.2f}".ljust(max_width + 2) + " |")
        self._log(f"| Maximum {unit}: {max_ops:.2f}".ljust(max_width + 2) + " |")
        self._log(f"| Minimum {unit}: {min_ops:.2f}".ljust(max_width + 2) + " |")
        
        # Divider line
        self._log("|" + "-" * (max_width + 2) + "|")
        
        # Core ranking
        self._log(f"| {'CORE PERFORMANCE RANKING (Higher is Better)'.center(max_width)} |")
        self._log("|" + "-" * (max_width + 2) + "|")
        
        # Rank cores by performance
        ranked_cores = sorted(
            [(core_id, data['operations_per_second']) for core_id, data in results.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # List all cores ranked by performance
        for i, (logical_core_id, ops) in enumerate(ranked_cores):
            rank = i + 1
            physical_core_id = logical_core_id // 2  # Convert logical to physical core ID
            deviation_pct = (ops/avg_ops - 1) * 100
            # Simplified core display format:
            line = f"Core {physical_core_id}: {ops:.2f} {unit}  #{rank}  ({deviation_pct:+.2f}% from mean)"
            self._log(f"| {line.ljust(max_width)} |")
        
        # Check for cores with performance deviation > 20% from mean
        deviation_threshold = 0.20  # 20%
        unstable_cores = []
        
        for logical_core_id, result in results.items():
            ops = result['operations_per_second']
            physical_core_id = logical_core_id // 2
            deviation = abs(ops - avg_ops) / avg_ops
            
            if deviation > deviation_threshold:
                deviation_percent = deviation * 100
                difference = "higher" if ops > avg_ops else "lower"
                unstable_cores.append((physical_core_id, deviation_percent, difference))
        
        # Divider line
        self._log("|" + "-" * (max_width + 2) + "|")
        
        if unstable_cores:
            self._log(f"| {'⚠️ WARNING: POTENTIAL CORE INSTABILITY DETECTED'.center(max_width)} |")
            for physical_core_id, deviation, difference in unstable_cores:
                msg = f"Core {physical_core_id}: {deviation:.1f}% {difference} than average"
                self._log(f"| {msg.ljust(max_width)} |")
            self._log("|" + "-" * (max_width + 2) + "|")
        else:
            self._log(f"| {'✅ All cores within 20% of mean performance'.center(max_width)} |")

    def _summarize_multithreaded_results(self, max_width):
        """Summarize results for multi-threaded tests."""
        mt_results = self.results.get('multithreaded', {})
        if not mt_results:
            return
        
        # Divider line
        self._log("|" + "-" * (max_width + 2) + "|")
        
        # Header for multi-threaded tests
        self._log(f"| {'MULTI-THREADED TEST RESULTS'.center(max_width)} |")
        
        # Divider line
        self._log("|" + "-" * (max_width + 2) + "|")
        
        # Process each test type with consistent formatting
        test_types = [
            ('light', 'Light Load Multi-Threaded Test:', 'ops/sec'),
            ('heavy', 'Heavy Load (AVX) Multi-Threaded Test:', 'vector ops/sec'),
            ('sse-heavy', 'SSE-Heavy Load Multi-Threaded Test:', 'vector ops/sec'),
            ('avx-heavy', 'AVX-Heavy Load Multi-Threaded Test:', 'vector ops/sec')
        ]
        
        for test_type, test_header, ops_unit in test_types:
            if test_type in mt_results:
                result = mt_results[test_type]
                self._log(f"| {test_header.center(max_width)} |")
                self._log(f"| Threads: {result['thread_count']}".ljust(max_width + 2) + " |")
                
                # Handle different result formats consistently
                if 'total_iterations' in result:
                    self._log(f"| Total operations: {result['total_iterations']:,}".ljust(max_width + 2) + " |")
                elif 'total_operations' in result:
                    self._log(f"| Total operations: {result['total_operations']:,}".ljust(max_width + 2) + " |")
                elif 'thread_results' in result:
                    total_ops = sum(r.get('total_ops', r.get('iterations', 0)) for r in result['thread_results'])
                    self._log(f"| Total vector operations: {total_ops:,}".ljust(max_width + 2) + " |")
                
                self._log(f"| Time: {result['elapsed_seconds']:.2f} seconds".ljust(max_width + 2) + " |")
                self._log(f"| Overall performance: {result['operations_per_second']:,.2f} {ops_unit}".ljust(max_width + 2) + " |")
                self._log(f"| Per-thread average: {result['operations_per_second']/result['thread_count']:,.2f} {ops_unit}".ljust(max_width + 2) + " |")
                
                # Calculate scaling efficiency if we have single-core results
                if any(k != 'multithreaded' for k in self.results.keys()):
                    single_core_results = [data[test_type]['operations_per_second'] 
                                        for core_id, data in self.results.items() 
                                        if core_id != 'multithreaded' and test_type in data]
                    if single_core_results:
                        single_core_avg = sum(single_core_results) / len(single_core_results)
                        scaling_ratio = result['operations_per_second'] / (single_core_avg * result['thread_count'])
                        self._log(f"| Scaling efficiency: {scaling_ratio*100:.1f}%".ljust(max_width + 2) + " |")
                
                self._log("|" + "-" * (max_width + 2) + "|")