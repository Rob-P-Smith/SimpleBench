import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys
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
        self.root.title("CPU Core Benchmark")
        self.root.geometry("800x600")
        
        self.benchmark = CPUBenchmark()
        self.benchmark_thread = None
        self.is_running = False
        
        # Initialize graph handler
        self.graphs = BenchmarkGraphs(root)
        
        # Test selection variables
        self.benchmark_type = tk.StringVar(value="light")  # Default to light benchmark
        self.run_multicore_tests = tk.BooleanVar(value=True)
        
        # New variables for SSE and AVX benchmarks
        self.run_sse_heavy_tests = tk.BooleanVar(value=False)
        self.run_avx_heavy_tests = tk.BooleanVar(value=False)

        # New options for graphs and logging
        self.display_graphs = tk.BooleanVar(value=True)
        self.log_results = tk.BooleanVar(value=False)
        
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
        self.duration_var = tk.StringVar(value="1")
        
        # Create validation command
        vcmd = (self.root.register(self._validate_duration), '%P')
        
        # Entry widget with validation
        self.duration_entry = ttk.Entry(duration_frame, width=5, textvariable=self.duration_var, validate='key', validatecommand=vcmd)
        self.duration_entry.pack(side=tk.LEFT)
        
        # Create benchmark configuration frame (renamed from Test Selection)
        config_frame = ttk.LabelFrame(self.root, text="Benchmark Configuration", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Configure grid layout for the config frame
        config_frame.columnconfigure(0, weight=1)  # Left column
        config_frame.columnconfigure(1, weight=1)  # Right column
        
        # Create benchmark type radio buttons frame - place on left side
        benchmark_type_frame = ttk.LabelFrame(config_frame, text="Benchmark Type", padding="5")
        benchmark_type_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        # Create radio buttons for benchmark types
        ttk.Radiobutton(benchmark_type_frame, text="SSE Simple Load (Basic Operations)", 
                    variable=self.benchmark_type, value="light").pack(anchor=tk.W, padx=10, pady=2)
                    
        # Add new radio button for heavy tests (replacing the checkbox)
        ttk.Radiobutton(benchmark_type_frame, text="AVX Simple Load (Basic AVX Operations)", 
                    variable=self.benchmark_type, value="heavy").pack(anchor=tk.W, padx=10, pady=2)

        ttk.Radiobutton(benchmark_type_frame, text="SSE-Heavy Load (SSE2 Vector Operations)", 
                    variable=self.benchmark_type, value="sse-heavy").pack(anchor=tk.W, padx=10, pady=2)

        # Enable AVX option
        ttk.Radiobutton(benchmark_type_frame, text="AVX-Heavy Load (AVX Vector Operations)", 
                    variable=self.benchmark_type, value="avx-heavy").pack(anchor=tk.W, padx=10, pady=2)
        
        # Additional options frame - place on right side
        options_frame = ttk.LabelFrame(config_frame, text="Additional Options", padding="5")
        options_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        # Multi-core test checkbox
        multicore_check = ttk.Checkbutton(options_frame, text="Run Multi-Core Tests", 
                                         variable=self.run_multicore_tests)
        multicore_check.pack(anchor=tk.W, padx=10, pady=2)
        
        # Display graphs checkbox
        display_graphs_check = ttk.Checkbutton(options_frame, text="Display Graphs", 
                                         variable=self.display_graphs)
        display_graphs_check.pack(anchor=tk.W, padx=10, pady=2)
        
        # Log results checkbox
        log_results_check = ttk.Checkbutton(options_frame, text="Log Results", 
                                         variable=self.log_results)
        log_results_check.pack(anchor=tk.W, padx=10, pady=2)
        
        # Add some empty space for balance (in case more options are added later)
        ttk.Label(options_frame, text="").pack(pady=5)
        
        # Create results view
        self.results_frame = ttk.Frame(self.root, padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_view = ResultsView(self.results_frame)
    
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
        
        # Check benchmark selection
        benchmark_type = self.benchmark_type.get()
        
        # Set benchmark flags based on radio selection
        run_light_tests = benchmark_type == "light"
        run_heavy_tests = benchmark_type == "heavy"  # Updated to use radio button value
        run_sse_heavy_tests = benchmark_type == "sse-heavy"
        run_avx_heavy_tests = benchmark_type == "avx-heavy"
        
        # Make sure at least one test type is selected
        if not any([run_light_tests, run_heavy_tests, run_sse_heavy_tests, run_avx_heavy_tests]):
            messagebox.showerror("Invalid Selection", "Please select at least one test type.")
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
        
        # Get multi-core test option
        run_multicore = self.run_multicore_tests.get()
        enable_logging = self.log_results.get()
        
        # Display selected tests
        self.results_view.clear()
        self.results_view.add_message(f"Starting benchmark with {duration} seconds per test...")
        self.results_view.add_message(f"Tests selected:")
        
        # Show selected benchmark type
        if run_light_tests:
            self.results_view.add_message(f"  - SSE Simple Load: Basic operations to measure baseline performance")
        elif run_heavy_tests:
            self.results_view.add_message(f"  - AVX Simple Load: Basic AVX operations to stress the CPU")
        elif run_sse_heavy_tests:
            self.results_view.add_message(f"  - SSE-Heavy Load: Vector operations using SSE2 instructions")
        elif run_avx_heavy_tests:
            self.results_view.add_message(f"  - AVX-Heavy Load: Vector operations using AVX instructions")
            
        # Show if multi-core tests are selected
        if run_multicore:
            self.results_view.add_message(f"  - Multi-threaded tests using all logical cores")
        else:
            self.results_view.add_message(f"  - Single-core tests only")
            
        self.results_view.add_message(f"Testing even-numbered logical cores only (one thread per physical core)")
        self.results_view.add_message("")

        def run_benchmark():
            try:
                self.benchmark.start(
                    progress_callback=self._update_progress,
                    completed_callback=self._benchmark_completed,
                    duration_per_core=duration,  # Pass the duration to the benchmark
                    run_light_tests=run_light_tests,
                    run_heavy_tests=run_heavy_tests,
                    run_sse_heavy_tests=run_sse_heavy_tests,
                    run_avx_heavy_tests=run_avx_heavy_tests,
                    run_multicore_tests=run_multicore,
                    enable_logging=enable_logging 
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
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write(self.results_view.get_all_text())
            
            messagebox.showinfo("Save Successful", f"Results saved to {os.path.abspath(filename)}")
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
        
        # Only show results window if display_graphs is checked
        if self.display_graphs.get() and hasattr(self.benchmark, 'results') and self.benchmark.results:
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

# Add this at the bottom of the file to allow direct execution
if __name__ == "__main__":
    # Configure path for imports when run directly
    if __package__ is None:
        import sys
        from pathlib import Path
        file_path = Path(__file__).resolve()
        parent = file_path.parent.parent.parent
        sys.path.append(str(parent))
        
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()