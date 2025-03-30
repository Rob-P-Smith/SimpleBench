def save_results_to_file(results, file_path):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

def load_results_from_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        results = [line.strip() for line in file.readlines()]
    return results