from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class BenchmarkGraphs:
    """Class for visualizing benchmark results as graphs using collected benchmark data."""
    
    def __init__(self, parent):
        """Initialize with parent Tkinter window."""
        self.parent = parent
        self.results_window = None
        self.graph_frames = {}
        
    def close_window(self):
        """Close the results window if it exists."""
        if self.results_window and self.results_window.winfo_exists():
            self.results_window.destroy()
            self.results_window = None
    
    def show_results(self, benchmark_results):
        """Create a new window showing benchmark results with graphs."""
        # Close any existing window
        self.close_window()
        
        # Create new window
        self.results_window = tk.Toplevel(self.parent)
        self.results_window.title("CPU Benchmark Results")
        self.results_window.geometry("1000x800")
        self.results_window.minsize(800, 600)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Track which test types have data
        has_light_data = False
        has_heavy_data = False
        has_sse_heavy_data = False
        has_avx_heavy_data = False
        
        # Check if we have progress data in results
        for core_id, data in benchmark_results.items():
            if core_id != 'multithreaded':
                if 'light' in data and 'progress' in data['light']:
                    has_light_data = True
                if 'heavy' in data and 'progress' in data['heavy']:
                    has_heavy_data = True
                if 'sse-heavy' in data and 'progress' in data['sse-heavy']:
                    has_sse_heavy_data = True
                if 'avx-heavy' in data and 'progress' in data['avx-heavy']:
                    has_avx_heavy_data = True
        
        # Only create tabs for tests that have data
        if has_light_data:
            light_tab = ttk.Frame(notebook)
            notebook.add(light_tab, text="SSE Simple Test")
            self._create_performance_graph(light_tab, benchmark_results, 'light', 
                                          "SSE Simple Load Performance Over Time")
            
        if has_heavy_data:
            heavy_tab = ttk.Frame(notebook)
            notebook.add(heavy_tab, text="AVX Simple Test")
            self._create_performance_graph(heavy_tab, benchmark_results, 'heavy', 
                                          "AVX Simple Load Performance Over Time")
            
        if has_sse_heavy_data:
            sse_heavy_tab = ttk.Frame(notebook)
            notebook.add(sse_heavy_tab, text="SSE-Heavy Test")
            self._create_performance_graph(sse_heavy_tab, benchmark_results, 'sse-heavy', 
                                          "SSE-Heavy Load Performance Over Time")
            
        if has_avx_heavy_data:
            avx_heavy_tab = ttk.Frame(notebook)
            notebook.add(avx_heavy_tab, text="AVX-Heavy Test")
            self._create_performance_graph(avx_heavy_tab, benchmark_results, 'avx-heavy', 
                                          "AVX-Heavy Load Performance Over Time")
    
    def _create_performance_graph(self, parent_frame, benchmark_results, test_type, title):
        """Create a performance graph for a specific test type using collected data."""
        frame = ttk.Frame(parent_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Determine if we need to split the display into graph and summary
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=4)  # Graph takes 80% of height
        frame.rowconfigure(1, weight=1)  # Summary takes 20% of height
        
        # Create a frame for the graph
        graph_frame = ttk.Frame(frame)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create a frame for the summary text
        summary_frame = ttk.Frame(frame)
        summary_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create matplotlib figure for the graph
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Collect core IDs that have data
        cores = []
        for core_id, data in benchmark_results.items():
            if core_id != 'multithreaded' and test_type in data and 'progress' in data[test_type]:
                cores.append(core_id)
        
        # Skip if no cores have data
        if not cores:
            label = ttk.Label(graph_frame, text="No progress data available for plotting")
            label.pack(pady=20)
            return
            
        # Sort cores for consistent coloring
        cores = sorted(cores)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cores)))
        
        # Calculate overall mean performance for y-axis centering
        all_perf_points = []
        core_data = {}  # Store data for each core for later plotting
        
        # First pass: collect all performance data
        for i, core_id in enumerate(cores):
            physical_core_id = core_id // 2  # Convert logical to physical core ID
            
            # Get progress data from results
            progress_data = benchmark_results[core_id][test_type]['progress']
            
            # Extract time points and performance values
            time_points = []
            perf_points = []
            
            for point in progress_data:
                if 'elapsed_seconds' in point and 'operations_per_second' in point:
                    time_points.append(point['elapsed_seconds'])
                    perf_points.append(point['operations_per_second'])
            
            # Only include if we have data
            if time_points and perf_points:
                all_perf_points.extend(perf_points)
                core_data[core_id] = {
                    'physical_core_id': physical_core_id,
                    'time_points': time_points,
                    'perf_points': perf_points,
                    'color': colors[i]
                }
        
        # Calculate mean and limits for the y-axis
        if all_perf_points:
            mean_performance = np.mean(all_perf_points)
            deviation_threshold = 0.20  # 20%
            
            # Calculate y-axis limits based on mean ±20%
            y_min = mean_performance * (1 - deviation_threshold)
            y_max = mean_performance * (1 + deviation_threshold)
            
            # Check if any points fall outside the ±20% range
            all_min = min(all_perf_points)
            all_max = max(all_perf_points)
            
            # Extend range if needed to include outliers, with 5% padding
            if all_min < y_min:
                y_min = all_min * 0.95
            if all_max > y_max:
                y_max = all_max * 1.05
                
            # Second pass: plot the data with the calculated limits
            for core_id, data in core_data.items():
                physical_core_id = data['physical_core_id']
                time_points = data['time_points']
                perf_points = data['perf_points']
                color = data['color']
                
                # Plot the main line
                line, = ax.plot(time_points, perf_points, 
                        label=f"Core {physical_core_id}", 
                        color=color, 
                        marker='o', 
                        markersize=3)
                        
                # Add core number label directly on the line
                # Use the midpoint of the line to place the label
                if len(time_points) > 0:
                    midpoint_idx = len(time_points) // 2
                    if midpoint_idx < len(time_points):
                        ax.text(time_points[midpoint_idx], perf_points[midpoint_idx], 
                            f" Core {physical_core_id}", 
                            color=color, 
                            fontweight='bold')

                # Highlight points outside the acceptable range with red circles
                outlier_times = []
                outlier_perfs = []
                
                for t, p in zip(time_points, perf_points):
                    if p < mean_performance * (1 - deviation_threshold) or p > mean_performance * (1 + deviation_threshold):
                        outlier_times.append(t)
                        outlier_perfs.append(p)
                
                if outlier_times:
                    ax.scatter(outlier_times, outlier_perfs, s=80, facecolors='none', 
                              edgecolors='red', linewidths=2, alpha=0.7,
                              label=f"Core {physical_core_id} Outliers" if i == 0 else "")
            
            # Add horizontal lines for mean and ±20% thresholds
            ax.axhline(y=mean_performance, color='green', linestyle='-', alpha=0.7, label="Mean")
            ax.axhline(y=mean_performance * (1 + deviation_threshold), color='red', linestyle='--', alpha=0.5, label="+20% Threshold")
            ax.axhline(y=mean_performance * (1 - deviation_threshold), color='red', linestyle='--', alpha=0.5, label="-20% Threshold")
            
            # Set y-axis limits
            ax.set_ylim(y_min, y_max)
            
            # Add acceptable range shading
            ax.axhspan(mean_performance * (1 - deviation_threshold), 
                    mean_performance * (1 + deviation_threshold), 
                    alpha=0.1, color='green', label="Acceptable Range")
            
            # Set labels and title
            ax.set_xlabel('Time (seconds)')
            
            # Set appropriate y-axis label based on test type
            if test_type == 'light':
                ax.set_ylabel('Operations per Second')
            elif test_type == 'heavy':
                ax.set_ylabel('Vector Operations per Second')
            elif test_type == 'sse-heavy':
                ax.set_ylabel('SSE Vector Operations per Second')
            elif test_type == 'avx-heavy':
                ax.set_ylabel('AVX Vector Operations per Second')
            else:
                ax.set_ylabel('Operations per Second')
                
            ax.set_title(f"{title}\nMean: {mean_performance:.2f} ops/sec")
            ax.grid(True)
            
            # Add legend with custom handler for the shaded area
            handles, labels = ax.get_legend_handles_labels()
            
            # Reduce duplicate labels in the legend (keep one for each core plus the mean lines)
            unique_labels = []
            unique_handles = []
            seen_labels = set()
            
            for handle, label in zip(handles, labels):
                if label not in seen_labels or label.startswith("Core "):
                    unique_labels.append(label)
                    unique_handles.append(handle)
                    seen_labels.add(label)
            
            ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 1))
            
            # Adjust layout for legend
            fig.tight_layout(rect=[0, 0, 0.85, 1])
            
            # Create canvas and display
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add the summary text below the graph
            self._create_summary_for_test(summary_frame, benchmark_results, test_type)
            
            self.graph_frames[test_type] = frame
        else:
            # No data points available
            label = ttk.Label(graph_frame, text="No performance data available for plotting")
            label.pack(pady=20)
    
    def _create_summary_for_test(self, frame, benchmark_results, test_type):
        """Create a summary display for a specific test type"""
        # Create text widget to display results
        results_text = tk.Text(frame, wrap=tk.WORD, font=('Consolas', 10), height=10)
        scrollbar = ttk.Scrollbar(frame, command=results_text.yview)
        results_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Format and display single core results
        test_results = {}
        for core_id, data in benchmark_results.items():
            if core_id != 'multithreaded' and test_type in data:
                test_results[core_id] = data[test_type]
        
        if test_results:
            # Calculate statistics
            ops_per_sec_values = [r['operations_per_second'] for r in test_results.values()]
            avg_ops = sum(ops_per_sec_values) / len(test_results)
            max_ops = max(ops_per_sec_values)
            min_ops = min(ops_per_sec_values)
            
            # Set appropriate unit based on test type
            ops_unit = "ops/sec"
            if test_type == 'heavy':
                ops_unit = "vector ops/sec"
            elif test_type == 'sse-heavy':
                ops_unit = "SSE vector ops/sec"
            elif test_type == 'avx-heavy':
                ops_unit = "AVX vector ops/sec"
            
            # Display statistics
            title = ""
            if test_type == 'light':
                title = "SSE SIMPLE LOAD TEST SUMMARY"
            elif test_type == 'heavy':
                title = "AVX SIMPLE LOAD TEST SUMMARY"
            elif test_type == 'sse-heavy':
                title = "SSE-HEAVY LOAD TEST SUMMARY"
            elif test_type == 'avx-heavy':
                title = "AVX-HEAVY LOAD TEST SUMMARY"
                
            results_text.insert(tk.END, f"{title}\n\n", "heading")
            results_text.tag_configure("heading", font=('Consolas', 10, 'bold'))
            
            results_text.insert(tk.END, f"Average {ops_unit} per core: {avg_ops:.2f}\n")
            results_text.insert(tk.END, f"Maximum {ops_unit}: {max_ops:.2f}\n")
            results_text.insert(tk.END, f"Minimum {ops_unit}: {min_ops:.2f}\n\n")
            
            # Core performance ranking
            results_text.insert(tk.END, "CORE PERFORMANCE RANKING (Higher is Better)\n", "subheading")
            results_text.tag_configure("subheading", font=('Consolas', 10, 'bold'))
            
            ranked_cores = sorted(
                [(core_id, data['operations_per_second']) for core_id, data in test_results.items()],
                key=lambda x: x[1], 
                reverse=True
            )
            
            for i, (logical_core_id, ops) in enumerate(ranked_cores):
                rank = i + 1
                physical_core_id = logical_core_id // 2  # Convert logical to physical core ID
                deviation_pct = (ops/avg_ops - 1) * 100
                line = f"Core {physical_core_id}: {ops:.2f} {ops_unit}  #{rank}  ({deviation_pct:+.2f}% from mean)\n"
                results_text.insert(tk.END, line)
            
            # Check for cores with performance deviation > 20%
            deviation_threshold = 0.20  # 20%
            unstable_cores = []
            
            for logical_core_id, result in test_results.items():
                ops = result['operations_per_second']
                physical_core_id = logical_core_id // 2
                deviation = abs(ops - avg_ops) / avg_ops
                
                if deviation > deviation_threshold:
                    unstable_cores.append((physical_core_id, deviation, (ops - avg_ops)))
            
            results_text.insert(tk.END, "\n")
            
            if unstable_cores:
                results_text.insert(tk.END, "⚠️ POTENTIAL INSTABILITY DETECTED\n", "warning")
                results_text.tag_configure("warning", foreground="red", font=('Consolas', 10, 'bold'))
                for physical_core_id, deviation, difference in unstable_cores:
                    deviation_pct = deviation * 100
                    direction = "above" if difference > 0 else "below"
                    results_text.insert(tk.END, f"Core {physical_core_id} is {deviation_pct:.1f}% {direction} the mean\n")
            else:
                results_text.insert(tk.END, "✅ All cores within 20% of mean performance\n", "good")
                results_text.tag_configure("good", foreground="green", font=('Consolas', 10, 'bold'))
        
        # Add multithreaded results if available
        if 'multithreaded' in benchmark_results and test_type in benchmark_results['multithreaded']:
            mt_data = benchmark_results['multithreaded'][test_type]
            results_text.insert(tk.END, f"\nMULTI-THREADED {test_type.upper()} TEST\n", "subheading")
            results_text.insert(tk.END, f"Threads: {mt_data.get('thread_count', 'N/A')}\n")
            results_text.insert(tk.END, f"Total operations: {mt_data.get('total_operations', mt_data.get('total_iterations', 'N/A')):,}\n")
            results_text.insert(tk.END, f"Time: {mt_data.get('elapsed_seconds', 'N/A'):.2f} seconds\n")
            results_text.insert(tk.END, f"Overall performance: {mt_data.get('operations_per_second', 'N/A'):,.2f} {ops_unit}\n")
        
        # Make text read-only
        results_text.configure(state=tk.DISABLED)