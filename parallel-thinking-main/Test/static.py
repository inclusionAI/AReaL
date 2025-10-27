import json

def calculate_averages(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize lists to store non-zero values
    number_of_threads_values = []
    max_width_values = []
    words = []
    # Extract non-zero values
    for item in data:
        if 'metrics' in item and item['metrics']['number_of_threads'] != 0:
            number_of_threads_values.append(item['metrics']['number_of_threads'])
        
        if 'metrics' in item and item['metrics']['max_width'] != 0:
            max_width_values.append(item['metrics']['max_width'])
        if 'metrics' in item and item['metrics']['word_count'] != 0:
            words.append(item['metrics']['word_count'])
    # Calculate averages
    avg_threads = sum(number_of_threads_values) / len(number_of_threads_values) if number_of_threads_values else 0
    avg_max_width = sum(max_width_values) / len(max_width_values) if max_width_values else 0
    avg_words = sum(words) / len(words) if words else 0
    return avg_threads, avg_max_width, len(number_of_threads_values), len(max_width_values), avg_words

# Usage
file_path = 'parallel_thinking_test_results.json'
avg_threads, avg_max_width, threads_count, width_count, avg_words = calculate_averages(file_path)

print(f"Average number_of_threads (excluding zeros): {avg_threads:.2f}")
print(f"Number of non-zero thread values: {threads_count}")
print(f"Average max_width (excluding zeros): {avg_max_width:.2f}")
print(f"Number of non-zero max_width values: {width_count}")
print(f"Average words (excluding zeros): {avg_words:.2f}")