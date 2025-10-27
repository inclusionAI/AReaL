def count_tokens_like_test(text):
    """
    Count tokens in the same way as used in test_parallel_thinking_performance.py
    This uses simple whitespace splitting as an approximation
    """
    # Split by whitespace - same as in the original code
    tokens = text.split()
    token_count = len(tokens)
    
    return token_count

def count_tokens_from_file(file_path):
    """
    Count tokens from a text file using the same method
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        token_count = count_tokens_like_test(text)
        return token_count, text
    
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return 0, ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0, ""

# Example usage
def main():
    # Method 1: Count tokens from a string
    sample_text = "This is a sample text with some words and tokens."
    token_count = count_tokens_like_test(sample_text)
    print(f"Token count for sample text: {token_count}")
    print(f"Text: {sample_text}")
    
    # Method 2: Count tokens from a file
    file_path = "reasoning.txt"  # Replace with your file path
    token_count, file_content = count_tokens_from_file(file_path)
    
    if token_count > 0:
        print(f"\nToken count for file '{file_path}': {token_count}")
        print(f"First 200 characters: {file_content[:200]}...")
    
    # Method 3: Interactive input
    print(f"\nEnter text to count tokens (or 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        
        token_count = count_tokens_like_test(user_input)
        print(f"Token count: {token_count}")

if __name__ == "__main__":
    main()