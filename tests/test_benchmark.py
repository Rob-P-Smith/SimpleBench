import unittest
from src.benchmark.cpu_benchmark import CpuBenchmark

class TestCpuBenchmark(unittest.TestCase):

    def setUp(self):
        self.benchmark = CpuBenchmark()

    def test_start_benchmark(self):
        self.benchmark.start()
        self.assertTrue(self.benchmark.is_running)

    def test_stop_benchmark(self):
        self.benchmark.start()
        self.benchmark.stop()
        self.assertFalse(self.benchmark.is_running)

    def test_collect_data(self):
        self.benchmark.start()
        data = self.benchmark.collect_data()
        self.assertIsInstance(data, dict)  # Assuming data is collected as a dictionary
        self.benchmark.stop()

    def test_benchmark_results(self):
        self.benchmark.start()
        self.benchmark.stop()
        results = self.benchmark.get_results()
        self.assertIsNotNone(results)  # Ensure results are not None
        self.assertIn('operations_per_second', results)  # Check for a specific metric

if __name__ == '__main__':
    unittest.main()