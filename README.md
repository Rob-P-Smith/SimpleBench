# CPU Benchmark Application

A sophisticated CPU benchmarking tool designed to test the maximum computational performance of individual cores and full multi-threaded capability. The application uses specialized SSE2 integer operations for light tests and AVX vector operations for heavy tests, providing a comprehensive analysis of CPU capabilities.

## Features

- **Individual Core Benchmarking**: Tests each physical core for both light and heavy computational workloads
- **Multi-Threaded Testing**: Evaluates full CPU performance using all available logical processors
- **Real-time Priority Execution**: Runs tests at real-time priority to minimize system interference
- **Detailed Performance Metrics**:
  - Per-core performance ranking
  - Multi-threaded scaling efficiency
  - Identification of potential core instability
- **Intel and AMD Support**: Optimized for both Intel and AMD processors
- **High-Precision Timing**: Uses Windows HPET (High Precision Event Timer) when available
- **User-Friendly GUI**: Simple interface for configuring tests and viewing results
- **Results Export**: Save benchmark results to text files for later analysis

## Technical Details

### Benchmark Types

1. **Light Load Test (SSE2 Integer Operations)**
   - Uses 256-element integer arrays optimized for L1 cache
   - Performs integer addition, multiplication, subtraction, and bitwise operations
   - Evaluates basic computational throughput using SIMD instructions

2. **Heavy Load Test (AVX Vector Operations)**
   - Uses 4000-element floating-point arrays
   - Performs complex transcendental operations (sin, cos, exp, log, sqrt, etc.)
   - Stresses CPU vector units, memory bandwidth, and thermal management

3. **Multi-Threaded Tests**
   - Runs both light and heavy tests with threads equal to logical processor count
   - Compares multi-threaded performance against single-core results
   - Calculates scaling efficiency to evaluate parallelism

## Project Structure

```
cpu-benchmark-app
├── src
│   ├── main.py               # Application entry point
│   ├── benchmark
│   │   ├── __init__.py       # Package initialization
│   │   ├── cpu_benchmark.py  # Main benchmark orchestrator
│   │   ├── light_benchmark.py # Light load benchmark implementation
│   │   ├── heavy_benchmark.py # Heavy load benchmark implementation
│   │   └── perf_counters.py  # Windows performance counter access
│   ├── gui
│   │   ├── __init__.py       # Package initialization
│   │   ├── main_window.py    # Main application window
│   │   └── results_view.py   # Results display panel
│   └── utils
│       ├── __init__.py       # Package initialization
│       ├── hpet_timer.py     # High-precision timer implementation
│       └── file_handler.py   # Results file management
├── tests
│   ├── __init__.py           # Package initialization
│   ├── test_benchmark.py     # Benchmark functionality tests
│   └── test_utils.py         # Utility function tests
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd cpu-benchmark-app
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python src/main.py
   ```

2. Configure the benchmark duration (1-30 seconds per test per core)

3. Click "Start Benchmark" to begin the testing process:
   - The application will first test each physical core individually
   - Then it will run multi-threaded tests using all logical cores
   - Results will appear in real-time in the output window

4. After completion, click "Save Results" to save the detailed output to a timestamped text file

## System Requirements

- Windows 7 or later (64-bit recommended)
- Python 3.7 or newer
- Modern CPU with SSE2 support (all x86-64 CPUs)
- 4GB RAM minimum (8GB+ recommended for large multi-core systems)
- Administrator privileges (for setting real-time process priority)

## Technical Implementation

- **Threading**: Uses Python's threading module for parallel execution
- **Process Affinity**: Sets CPU affinity to isolate single-core tests
- **Process Priority**: Elevates to real-time priority for consistent results
- **Performance Measurement**: Uses performance counters and high-precision timers
- **SIMD Operations**: Leverages NumPy for SSE2/AVX vector operations
- **Error Handling**: Graceful recovery from test failures

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.