import unittest
from src.utils.hpet_timer import start_timer, stop_timer
from src.utils.file_handler import save_results, load_results

class TestUtils(unittest.TestCase):

    def test_start_timer(self):
        timer = start_timer()
        self.assertIsNotNone(timer)

    def test_stop_timer(self):
        timer = start_timer()
        elapsed_time = stop_timer(timer)
        self.assertGreater(elapsed_time, 0)

    def test_save_results(self):
        results = {'core_1': 100, 'core_2': 200}
        file_path = 'test_results.json'
        save_results(results, file_path)
        loaded_results = load_results(file_path)
        self.assertEqual(results, loaded_results)

    def test_load_results(self):
        results = {'core_1': 100, 'core_2': 200}
        file_path = 'test_results.json'
        save_results(results, file_path)
        loaded_results = load_results(file_path)
        self.assertEqual(results, loaded_results)

if __name__ == '__main__':
    unittest.main()