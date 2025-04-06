import os
from datetime import datetime

def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_logs_dir():
    """Get the path to the logs directory."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(parent_dir)
    logs_dir = os.path.join(project_root, "logs")
    ensure_dir(logs_dir)
    return logs_dir

def get_results_dir():
    """Get the path to the results directory."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(parent_dir)
    results_dir = os.path.join(project_root, "results")
    ensure_dir(results_dir)
    return results_dir

def create_log_file():
    """Create a new log file in the logs directory."""
    logs_dir = get_logs_dir()
    log_filename = f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    return os.path.join(logs_dir, log_filename)

def append_to_log_file(log_filename, message):
    """Append a message to the log file."""
    with open(log_filename, "a") as f:
        f.write(message + "\n")
        f.flush()

def log_test_start(log_filename, core_id, test_type):
    """Log the start of a test to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    core_desc = "multithreaded" if core_id == "multithreaded" else f"Core {core_id//2}"
    message = f"{timestamp} - Starting {test_type} test on {core_desc}"
    append_to_log_file(log_filename, message)

def save_results_to_file(results, filename=None):
    """Save results to a file in the results directory."""
    results_dir = get_results_dir()
    if filename is None:
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = os.path.join(results_dir, filename)
    
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
    
    return file_path

def load_results_from_file(filename):
    """Load results from a file in the results directory."""
    results_dir = get_results_dir()
    file_path = os.path.join(results_dir, filename)
    
    results = []
    with open(file_path, 'r') as file:
        results = [line.strip() for line in file.readlines()]
    
    return results