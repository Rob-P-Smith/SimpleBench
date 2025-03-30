from tkinter import ttk, messagebox
import tkinter as tk
import os
import sys
import threading
from datetime import datetime

# Add the parent directory to sys.path to enable imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.benchmark.cpu_benchmark import CPUBenchmark
from src.gui.results_view import ResultsView
from src.gui.graphs import BenchmarkGraphs

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("SimpleBench CPU Benchmark")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Initialize benchmark engine
        self.benchmark = CPUBenchmark()
        
        # Initialize graph module
        self.graphs = BenchmarkGraphs(root)
        
        # Create variables for checkboxes
        self.run_light_tests = tk.BooleanVar(value=True)
        self.run_heavy_tests = tk.BooleanVar(value=True)
        self.run_multicore_tests = tk.BooleanVar(value=True)
        
        # Flag to track if a benchmark is running
        self.is_running = False
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        # Create top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create buttons
        self.start_button = ttk.Button(control_frame, text="Start Benchmark", command=self._start_benchmark)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Benchmark", command=self._stop_benchmark, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="Save Results", command=self._save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Add view results button
        self.view_results_button = ttk.Button(control_frame, text="View Graphs", command=self._view_graphs, state=tk.DISABLED)
        self.view_results_button.pack(side=tk.LEFT, padx=5)
        
        # Add duration input field
        duration_frame = ttk.Frame(control_frame)
        duration_frame.pack(side=tk.RIGHT, padx=15)
        
        ttk.Label(duration_frame, text="Duration per core (1-30 sec):").pack(side=tk.LEFT, padx=5)
        
        # StringVar for validation
        self.duration_var = tk.StringVar(value="10")
        
        # Create validation command
        vcmd = (self.root.register(self._validate_duration), '%P')
        
        # Entry widget with validation
        self.duration_entry = ttk.Entry(duration_frame, width=5, textvariable=self.duration_var, validate='key', validatecommand=vcmd)
        self.duration_entry.pack(side=tk.LEFT)
        
        # Create test selection frame
        test_frame = ttk.LabelFrame(self.root, text="Test Selection", padding="10")
        test_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Test type selection
        test_type_frame = ttk.Frame(test_frame)
        test_type_frame.pack(fill=tk.X, pady=5)
        
        # Create "Run All Tests" button
        run_all_button = ttk.Button(test_type_frame, text="Run All Tests", 
                                    command=lambda: self._set_all_tests(True))
        run_all_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create "Clear All" button
        clear_all_button = ttk.Button(test_type_frame, text="Clear All", 
                                    command=lambda: self._set_all_tests(False))
        clear_all_button.pack(side=tk.LEFT, padx=(0, 20))
        
        # SSE test checkbox (formerly Light)
        light_check = ttk.Checkbutton(test_type_frame, text="SSE Tests", 
                                    variable=self.run_light_tests)
        light_check.pack(side=tk.LEFT, padx=10)
        
        # AVX test checkbox (formerly Heavy)
        heavy_check = ttk.Checkbutton(test_type_frame, text="AVX Tests", 
                                    variable=self.run_heavy_tests)
        heavy_check.pack(side=tk.LEFT, padx=10)
        
        # Multi-core test checkbox
        multicore_check = ttk.Checkbutton(test_type_frame, text="Multi-Core Tests", 
                                        variable=self.run_multicore_tests)
        multicore_check.pack(side=tk.LEFT, padx=10)
        
        # Create results view area
        # Create ResultsView instance
        self.results_view = ResultsView(self.root)
    
    def _set_all_tests(self, value):
        """Set all test selection checkboxes to the given value."""
        self.run_light_tests.set(value)
        self.run_heavy_tests.set(value)
        self.run_multicore_tests.set(value)
    
    def _validate_duration(self, value):
        """Validate that the entered duration is a number between 1 and 30."""
        if value == "":
            return True
        try:
            val = int(value)
            return 1 <= val <= 30
        except ValueError:
            return False
        
    def _start_benchmark(self):
        if self.is_running:
            return
        
        # Check if at least one test type is selected
        if not any([self.run_light_tests.get(), self.run_heavy_tests.get()]):
            messagebox.showerror("Invalid Selection", "Please select at least one test type (SSE or AVX).")
            return
            
        # Get the duration from the entry field
        try:
            duration = int(self.duration_var.get())
            if duration < 1 or duration > 30:
                messagebox.showerror("Invalid Input", "Duration must be between 1 and 30 seconds.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Duration must be a number.")
            return
            
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.view_results_button.config(state=tk.DISABLED)
        self.duration_entry.config(state=tk.DISABLED)
        
        # Close any existing graph window
        self.graphs.close_window()
        
        # Get test options
        run_light = self.run_light_tests.get()
        run_heavy = self.run_heavy_tests.get()
        run_multicore = self.run_multicore_tests.get()
        
        # Display selected tests
        self.results_view.clear()
        self.results_view.add_message(f"Starting benchmark with {duration} seconds per test...")
        self.results_view.add_message(f"Tests selected:")
        
        if run_light:
            self.results_view.add_message(f"  - SSE Tests: Simple integer operations to measure basic performance")
        if run_heavy:
            self.results_view.add_message(f"  - AVX Tests: Vector operations to stress the CPU")
        if run_multicore:
            self.results_view.add_message(f"  - Multi-threaded tests using all logical cores")
        else:
            self.results_view.add_message(f"  - Single-core tests only")
    
        def run_benchmark():
            try:
                self.benchmark.start(
                    progress_callback=self._update_progress,
                    completed_callback=self._benchmark_completed,
                    duration_per_core=duration,  # Pass the duration to the benchmark
                    run_light_tests=run_light,
                    run_heavy_tests=run_heavy,
                    run_multicore_tests=run_multicore
                )
            except Exception as err:
                # Use a local variable that won't cause closure problems
                error_message = str(err)
                self.root.after(0, lambda msg=error_message: self._handle_error(msg))
        
        self.benchmark_thread = threading.Thread(target=run_benchmark)
        self.benchmark_thread.daemon = True
        self.benchmark_thread.start()
        
    def _stop_benchmark(self):
        if not self.is_running:
            return
                
        self.benchmark.stop()
        self.results_view.add_message("Benchmark stopping...")
        
    def _save_results(self):
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(project_root, "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.results_view.get_all_text())
            
            messagebox.showinfo("Save Successful", f"Results saved to {os.path.abspath(filepath)}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Failed to save results: {str(e)}")
    
    def _view_graphs(self):
        """Show the benchmark result graphs in a new window."""
        try:
            if hasattr(self.benchmark, 'results') and self.benchmark.results:
                self.graphs.show_results(self.benchmark.results)
            else:
                messagebox.showinfo("No Data", "No benchmark results available to display.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display graphs: {str(e)}")
        
    def _update_progress(self, message):
        self.root.after(0, lambda: self.results_view.add_message(message))
        
    def _benchmark_completed(self):
        self.root.after(0, self._handle_completion)
        
    def _handle_completion(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.view_results_button.config(state=tk.NORMAL)  # Enable graph viewing
        self.duration_entry.config(state=tk.NORMAL)
        self.results_view.add_message("Benchmark completed.")
        
        # Automatically show the results window with graphs
        if hasattr(self.benchmark, 'results') and self.benchmark.results:
            self.graphs.show_results(self.benchmark.results)
        
    def _handle_error(self, error_message):
        """Handle benchmark errors."""
        self.results_view.add_message(f"ERROR: {error_message}")
        
        # Re-enable buttons and inputs
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.duration_entry.config(state=tk.NORMAL)
        
        messagebox.showerror("Benchmark Error", f"An error occurred: {error_message}")