def save_results_to_file(results, directory):
    import os
    from datetime import datetime

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.txt"
    file_path = os.path.join(directory, filename)

    # Save results to the file
    with open(file_path, 'w') as f:
        for result in results:
            f.write(f"{result}\n")

    return file_path

def load_results_from_file(file_path):
    results = []
    with open(file_path, 'r') as f:
        results = f.readlines()
    return [result.strip() for result in results]