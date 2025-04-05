import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
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
        
        # Create tabs for light and heavy tests
        light_tab = ttk.Frame(notebook)
        heavy_tab = ttk.Frame(notebook)
        summary_tab = ttk.Frame(notebook)
        
        notebook.add(light_tab, text="SSE Test Results")
        notebook.add(heavy_tab, text="AVX Test Results")
        notebook.add(summary_tab, text="Summary")
        
        # Extract progress data from benchmark results if available
        has_light_data = False
        has_heavy_data = False
        
        # Check if we have progress data in results
        for core_id, data in benchmark_results.items():
            if core_id != 'multithreaded':
                if 'light' in data and 'progress' in data['light']:
                    has_light_data = True
                if 'heavy' in data and 'progress' in data['heavy']:
                    has_heavy_data = True
        
        # Add performance graphs
        if has_light_data:
            self._create_performance_graph(light_tab, benchmark_results, 'light', "SSE Load Performance Over Time")
            
        if has_heavy_data:
            self._create_performance_graph(heavy_tab, benchmark_results, 'heavy', "AVX Load Performance Over Time")
        
        # Create summary display
        self._create_summary_display(summary_tab, benchmark_results)
    
    def _create_performance_graph(self, parent_frame, benchmark_results, test_type, title):
        """Create a performance graph for a specific test type using collected data."""
        frame = ttk.Frame(parent_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Collect core IDs that have data
        cores = []
        for core_id, data in benchmark_results.items():
            if core_id != 'multithreaded' and test_type in data and 'progress' in data[test_type]:
                cores.append(core_id)
        
        # Skip if no cores have data
        if not cores:
            label = ttk.Label(frame, text="No progress data available for plotting")
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
                    mid_idx = len(time_points) // 2
                    ax.text(time_points[mid_idx], perf_points[mid_idx], 
                            str(physical_core_id),  # Just the number
                            color=color, 
                            fontweight='bold',
                            ha='center', 
                            va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, pad=1))

                # Highlight points outside the acceptable range with red circles
                outlier_times = []
                outlier_perfs = []
                
                for t, p in zip(time_points, perf_points):
                    if p < mean_performance * (1 - deviation_threshold) or p > mean_performance * (1 + deviation_threshold):
                        outlier_times.append(t)
                        outlier_perfs.append(p)
                
                if outlier_times:
                    ax.plot(outlier_times, outlier_perfs, 'o', color='red', 
                            markersize=8, markerfacecolor='none', markeredgewidth=2)
            
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
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.graph_frames[test_type] = frame
        else:
            # No data points available
            label = ttk.Label(frame, text="No performance data available for plotting")
            label.pack(pady=20)
    
    def _create_summary_display(self, parent_frame, benchmark_results):
        """Create summary display with text results."""
        frame = ttk.Frame(parent_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text widget to display results
        results_text = tk.Text(frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(frame, command=results_text.yview)
        results_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Insert summary results
        if not benchmark_results:
            results_text.insert(tk.END, "No benchmark results available.")
            return
        
        # Display header
        results_text.insert(tk.END, "CPU BENCHMARK RESULTS SUMMARY\n\n")
        results_text.tag_configure("header", font=('Consolas', 12, 'bold'))
        results_text.tag_add("header", "1.0", "1.end")
        
        # Format and display single core results
        if any(core_id != 'multithreaded' for core_id in benchmark_results.keys()):
            # Light test results
            light_results = {}
            for core_id, data in benchmark_results.items():
                if core_id != 'multithreaded' and 'light' in data:
                    light_results[core_id] = data['light']
            
            if light_results:
                results_text.insert(tk.END, "LIGHT LOAD TEST RESULTS\n", "section")
                results_text.tag_configure("section", font=('Consolas', 11, 'bold'))
                self._format_test_results(results_text, light_results)
            
            # Heavy test results
            heavy_results = {}
            for core_id, data in benchmark_results.items():
                if core_id != 'multithreaded' and 'heavy' in data:
                    heavy_results[core_id] = data['heavy']
            
            if heavy_results:
                results_text.insert(tk.END, "\nHEAVY LOAD TEST RESULTS\n", "section")
                self._format_test_results(results_text, heavy_results)
        
        # Format and display multi-threaded results
        if 'multithreaded' in benchmark_results:
            mt_results = benchmark_results['multithreaded']
            
            results_text.insert(tk.END, "\nMULTI-THREADED TEST RESULTS\n", "section")
            
            if 'light' in mt_results:
                light = mt_results['light']
                results_text.insert(tk.END, "\nLight Load Multi-Threaded Test:\n", "subsection")
                results_text.tag_configure("subsection", font=('Consolas', 10, 'bold'))
                results_text.insert(tk.END, f"  Threads: {light['thread_count']}\n")
                results_text.insert(tk.END, f"  Total operations: {light['total_iterations']:,}\n")
                results_text.insert(tk.END, f"  Time: {light['elapsed_seconds']:.2f} seconds\n")
                results_text.insert(tk.END, f"  Overall performance: {light['operations_per_second']:,.2f} ops/sec\n")
            
            if 'heavy' in mt_results:
                heavy = mt_results['heavy']
                results_text.insert(tk.END, "\nHeavy Load Multi-Threaded Test:\n", "subsection")
                results_text.insert(tk.END, f"  Threads: {heavy['thread_count']}\n")
                results_text.insert(tk.END, f"  Time: {heavy['elapsed_seconds']:.2f} seconds\n")
                results_text.insert(tk.END, f"  Overall performance: {heavy['operations_per_second']:,.2f} ops/sec\n")
        
        # Make text read-only
        results_text.configure(state=tk.DISABLED)
    
    def _format_test_results(self, text_widget, results):
        """Format and add test results to the text widget."""
        if not results:
            return
            
        # Calculate statistics
        ops_per_sec_values = [r['operations_per_second'] for r in results.values()]
        avg_ops = sum(ops_per_sec_values) / len(results)
        max_ops = max(ops_per_sec_values)
        min_ops = min(ops_per_sec_values)
        
        # Display statistics
        text_widget.insert(tk.END, f"\nAverage ops/sec per core: {avg_ops:.2f}\n")
        text_widget.insert(tk.END, f"Maximum ops/sec: {max_ops:.2f}\n")
        text_widget.insert(tk.END, f"Minimum ops/sec: {min_ops:.2f}\n")
        
        # Rank cores by performance
        text_widget.insert(tk.END, "\nCORE PERFORMANCE RANKING (Higher is Better)\n", "subsection")
        
        ranked_cores = sorted(
            [(core_id, data['operations_per_second']) for core_id, data in results.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        for i, (logical_core_id, ops) in enumerate(ranked_cores):
            rank = i + 1
            physical_core_id = logical_core_id // 2  # Convert logical to physical core ID
            deviation_pct = (ops/avg_ops - 1) * 100
            line = f"Core {physical_core_id}: {ops:.2f} ops/sec  #{rank}  ({deviation_pct:+.2f}% from mean)\n"
            text_widget.insert(tk.END, line)
        
        # Check for cores with performance deviation > 20%
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
        
        if unstable_cores:
            text_widget.insert(tk.END, "\n⚠️ POTENTIAL INSTABILITY DETECTED\n", "warning")
            text_widget.tag_configure("warning", foreground="red", font=('Consolas', 10, 'bold'))
            for physical_core_id, deviation, difference in unstable_cores:
                line = f"Core {physical_core_id}: {deviation:.2f}% {difference} than average\n"
                text_widget.insert(tk.END, line)
        else:
            text_widget.insert(tk.END, "\n✅ All cores within 20% of mean performance\n", "good")
            text_widget.tag_configure("good", foreground="green", font=('Consolas', 10, 'bold'))