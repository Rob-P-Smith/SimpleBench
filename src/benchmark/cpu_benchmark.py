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
from src.utils.hpet_timer import HPETTimer

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
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Set log filename in logs directory
        self.log_filename = os.path.join(logs_dir, f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
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
    
    def _log(self, message):
        """Log a message using the progress callback if available."""
        print(message)  # Always print to console
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
          run_light_tests=True, run_heavy_tests=True, run_multicore_tests=True):
        """Start the benchmark process."""
        if self.running:
            return False
            
        self.running = True
        self._stop_event.clear()
        self.progress_callback = progress_callback
        self.completed_callback = completed_callback
        self.results = {}
        
        # Initialize log file with header
        try:
            with open(self.log_filename, "w") as f:
                f.write(f"CPU BENCHMARK LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 60 + "\n\n")
        except Exception as e:
            self._log(f"Error creating log file: {str(e)}")
        
        # Start benchmark in a separate thread
        self._benchmark_thread = threading.Thread(
            target=self._run_benchmark,
            args=(duration_per_core, run_light_tests, run_heavy_tests, run_multicore_tests)
        )
        self._benchmark_thread.daemon = True
        self._benchmark_thread.start()
        
        return True

    def _run_benchmark(self, duration_per_core, run_light_tests, run_heavy_tests, run_multicore_tests):
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
            
            # Run multi-threaded tests
            if run_multicore_tests:
                self._log("\nStarting multi-threaded tests using all cores")
                self._run_multicore_tests(duration_per_core, run_light_tests, run_heavy_tests)
            
            # Complete the benchmark
            self._log("\nAll benchmarks completed!")
            
            # Summarize results
            self._summarize_results()
        
        except Exception as e:
            self._log(f"Error during benchmark: {str(e)}")
        finally:
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
                
                if test_type == 'light':
                    # Light load test
                    self._log(f"Starting light load benchmark on Core {physical_core_id}...")
                    self.current_core = core_id
                    
                    # Set process affinity to this core
                    process = psutil.Process()
                    original_affinity = process.cpu_affinity()
                    
                    try:
                        # Set affinity to only use the specified core
                        process.cpu_affinity([core_id])
                        
                        # Run the test
                        light_result = self.light_benchmark.run_single_core_test(core_id, duration)
                        
                        # Initialize core results if needed
                        if core_id not in self.results:
                            self.results[core_id] = {}
                            
                        self.results[core_id]['light'] = light_result
                    finally:
                        # Restore original affinity
                        try:
                            process.cpu_affinity(original_affinity)
                        except Exception:
                            pass
                    
                elif test_type == 'heavy':
                    # Heavy load test
                    self._log(f"Starting heavy AVX load benchmark on Core {physical_core_id}...")
                    self.current_core = core_id
                    
                    # Set process affinity to this core
                    process = psutil.Process()
                    original_affinity = process.cpu_affinity()
                    
                    try:
                        # Set affinity to only use the specified core
                        process.cpu_affinity([core_id])
                        
                        # Run the test
                        heavy_result = self.heavy_benchmark.run_single_core_test(core_id, duration)
                        
                        # Initialize core results if needed
                        if core_id not in self.results:
                            self.results[core_id] = {}
                            
                        self.results[core_id]['heavy'] = heavy_result
                    finally:
                        # Restore original affinity
                        try:
                            process.cpu_affinity(original_affinity)
                        except Exception:
                            pass

    def _run_multicore_tests(self, duration, run_light_tests, run_heavy_tests):
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
            
        except Exception as e:
            self._log(f"Error in multi-threaded tests: {str(e)}")
        finally:
            self.current_core = None

    def stop(self):
        """Stop the benchmark process."""
        self._stop_event.set()
        self._log("Benchmark stopping...")
        return True

    def _summarize_results(self):
        """Summarize benchmark results."""
        if not self.results:
            self._log("No benchmark results to summarize")
            return
            
        try:
            # Check if we have valid results for light and heavy tests
            light_results = {}
            heavy_results = {}
            
            for core_id, data in self.results.items():
                if core_id == 'multithreaded':
                    continue  # Skip multithreaded results for now
                    
                if 'light' in data and isinstance(data['light'], dict) and 'operations_per_second' in data['light']:
                    light_results[core_id] = data['light']
                
                if 'heavy' in data and isinstance(data['heavy'], dict) and 'operations_per_second' in data['heavy']:
                    heavy_results[core_id] = data['heavy']
            
            # Calculate maximum width for formatting
            max_width = max(
                60,  # Increased minimum width
                len("CPU BENCHMARK RESULTS SUMMARY"),
                len("LIGHT LOAD TEST RESULTS (Simple Operations)"),
                len("HEAVY LOAD TEST RESULTS (AVX Vector Operations)"),
                len("MULTI-THREADED TEST RESULTS")
            )
            
            # Create the box with equals and pipes
            border = "=" * (max_width + 4)
            self._log("\n" + border)
            
            # Header
            title = "CPU BENCHMARK RESULTS SUMMARY"
            self._log(f"| {title.center(max_width)} |")
            
            # Summarize light load test results
            if light_results:
                self._summarize_test_results("LIGHT LOAD TEST RESULTS (Simple Operations)", 
                                        light_results, max_width)
            
            # Summarize heavy load test results
            if heavy_results:
                self._summarize_test_results("HEAVY LOAD TEST RESULTS (AVX Vector Operations)", 
                                        heavy_results, max_width)
            
            # Summarize multi-threaded test results if available
            if 'multithreaded' in self.results:
                self._summarize_multithreaded_results(max_width)
            
            # Close the box
            self._log(border)
            
            # Add log file location
            self._log(f"\nBenchmark log file: {os.path.abspath(self.log_filename)}")
        
        except Exception as e:
            self._log(f"Error summarizing results: {str(e)}")

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
        
        # Light load multi-threaded results
        if 'light' in mt_results:
            light = mt_results['light']
            self._log(f"| {'Light Load Multi-Threaded Test:'.center(max_width)} |")
            self._log(f"| Threads: {light['thread_count']}".ljust(max_width + 2) + " |")
            self._log(f"| Total operations: {light['total_iterations']:,}".ljust(max_width + 2) + " |")
            self._log(f"| Time: {light['elapsed_seconds']:.2f} seconds".ljust(max_width + 2) + " |")
            self._log(f"| Overall performance: {light['operations_per_second']:,.2f} ops/sec".ljust(max_width + 2) + " |")
            self._log(f"| Per-thread average: {light['operations_per_second']/light['thread_count']:,.2f} ops/sec".ljust(max_width + 2) + " |")
            
            # If we have single-core results, calculate scaling efficiency
            single_core_avg = 0
            if any(k != 'multithreaded' for k in self.results.keys()):
                single_core_results = [data['light']['operations_per_second'] 
                                    for core_id, data in self.results.items() 
                                    if core_id != 'multithreaded' and 'light' in data]
                if single_core_results:
                    single_core_avg = sum(single_core_results) / len(single_core_results)
                    scaling_ratio = light['operations_per_second'] / (single_core_avg * light['thread_count'])
                    self._log(f"| Scaling efficiency: {scaling_ratio*100:.1f}%".ljust(max_width + 2) + " |")
            
            self._log("|" + "-" * (max_width + 2) + "|")
        
        # Heavy load multi-threaded results
        if 'heavy' in mt_results:
            heavy = mt_results['heavy']
            self._log(f"| {'Heavy Load (AVX) Multi-Threaded Test:'.center(max_width)} |")
            self._log(f"| Threads: {heavy['thread_count']}".ljust(max_width + 2) + " |")
            total_ops = sum(r.get('total_ops', r.get('iterations', 0)) for r in heavy['thread_results'])
            self._log(f"| Total vector operations: {total_ops:,}".ljust(max_width + 2) + " |")
            self._log(f"| Time: {heavy['elapsed_seconds']:.2f} seconds".ljust(max_width + 2) + " |")
            self._log(f"| Overall performance: {heavy['operations_per_second']:,.2f} vector ops/sec".ljust(max_width + 2) + " |")
            self._log(f"| Per-thread average: {heavy['operations_per_second']/heavy['thread_count']:,.2f} vector ops/sec".ljust(max_width + 2) + " |")
            
            # If we have single-core results, calculate scaling efficiency
            single_core_avg = 0
            if any(k != 'multithreaded' for k in self.results.keys()):
                single_core_results = [data['heavy']['operations_per_second'] 
                                    for core_id, data in self.results.items() 
                                    if core_id != 'multithreaded' and 'heavy' in data]
                if single_core_results:
                    single_core_avg = sum(single_core_results) / len(single_core_results)
                    scaling_ratio = heavy['operations_per_second'] / (single_core_avg * heavy['thread_count'])
                    self._log(f"| Scaling efficiency: {scaling_ratio*100:.1f}%".ljust(max_width + 2) + " |")
    
    def _summarize_test_results(self, header, results, max_width):
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
        
        # Average performance
        self._log(f"| Average ops/sec per core: {avg_ops:.2f}".ljust(max_width + 2) + " |")
        self._log(f"| Maximum ops/sec: {max_ops:.2f}".ljust(max_width + 2) + " |")
        self._log(f"| Minimum ops/sec: {min_ops:.2f}".ljust(max_width + 2) + " |")
        
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
            line = f"Core {physical_core_id}: {ops:.2f} ops/sec  #{rank}  ({deviation_pct:+.2f}% from mean)"
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