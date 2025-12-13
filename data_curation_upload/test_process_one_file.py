import os
from process_steps_with_objectives import process_file

if __name__ == "__main__":
    # Test with one file
    INPUT_FILE = "/Users/zzy/Desktop/data_curation_new/result_merged/merged_steps_20251115_225055.txt"
    OUTPUT_DIR = "/Users/zzy/Desktop/data_curation_new/result_with_objectives_test"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process single file
    print("Testing with single file...")
    process_file(INPUT_FILE, OUTPUT_DIR)
    print("\nTest complete!")
