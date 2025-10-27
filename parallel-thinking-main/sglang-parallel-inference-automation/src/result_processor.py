def process_results(results_directory: str):
    import os
    import json
    from datetime import datetime

    def format_result(result):
        # Format the result as needed (e.g., as a JSON string)
        return json.dumps(result, indent=2)

    def save_result(result, timestamp: str):
        # Create a filename based on the timestamp
        filename = f"result_{timestamp}.json"
        filepath = os.path.join(results_directory, filename)

        # Save the formatted result to a file
        with open(filepath, 'w') as f:
            f.write(format_result(result))
        print(f"Result saved to {filepath}")

    def summarize_results(results):
        # Example summary logic (customize as needed)
        summary = {
            "total_runs": len(results),
            "successful_runs": sum(1 for r in results if r.get("success")),
            "failed_runs": sum(1 for r in results if not r.get("success")),
        }
        return summary

    # Ensure the results directory exists
    os.makedirs(results_directory, exist_ok=True)

    # Example results data (replace with actual results from inference runs)
    results = [
        {"run_id": 1, "success": True, "data": "Result data 1"},
        {"run_id": 2, "success": False, "error": "Error message"},
        # Add more results as needed
    ]

    # Process each result
    for result in results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_result(result, timestamp)

    # Generate a summary of the results
    summary = summarize_results(results)
    print("Summary of results:", summary)