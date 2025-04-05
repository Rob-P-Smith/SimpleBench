import os
from datetime import datetime

def ensure_directory_exists(directory_path):
    """Ensure that the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def get_project_root():
    """Get the project root directory (cpu-benchmark-app folder)."""
    # Starting from file_handler.py location, go up 2 directories instead of 3
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # This gets us to src folder
    parent_dir = os.path.dirname(current_dir)
    # This gets us to the cpu-benchmark-app folder (project root)
    project_root = os.path.dirname(parent_dir)
    return project_root

def get_logs_directory():
    """Get the path to the logs directory, creating it if necessary."""
    logs_dir = os.path.join(get_project_root(), "logs")
    return ensure_directory_exists(logs_dir)

def get_results_directory():
    """Get the path to the results directory, creating it if necessary."""
    results_dir = os.path.join(get_project_root(), "results")
    return ensure_directory_exists(results_dir)

def create_log_file():
    """Create a new log file with timestamp in the logs directory."""
    logs_dir = get_logs_directory()
    log_filename = f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(logs_dir, log_filename)
    
    # Create and initialize the log file
    with open(log_path, "w") as f:
        f.write(f"CPU BENCHMARK LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 60 + "\n\n")
    
    return log_path

def save_results_to_file(results, filename=None):
    """Save benchmark results to a file in the results directory."""
    results_dir = get_results_directory()
    
    if filename is None:
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    file_path = os.path.join(results_dir, filename)
    
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
    
    return file_path

def load_results_from_file(file_path):
    """Load benchmark results from a file."""
    results = []
    with open(file_path, 'r') as file:
        results = [line.strip() for line in file.readlines()]
    return results