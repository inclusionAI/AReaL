# SGLang Parallel Inference Automation

## Overview
This project implements a parallel inference automation system using the SGLang framework. It is designed to generate responses efficiently by running multiple inference processes concurrently. The results are stored in a structured manner, allowing for easy access and analysis.

## Project Structure
```
sglang-parallel-inference-automation
├── src
│   ├── sglang_inference.py        # Main logic for generating responses
│   ├── batch_runner.py            # Runs multiple inference instances
│   ├── result_processor.py         # Processes and analyzes results
│   └── utils
│       ├── __init__.py            # Initializes the utils package
│       ├── file_utils.py          # Utility functions for file operations
│       └── time_utils.py          # Utility functions for handling timestamps
├── config
│   ├── model_config.json           # Configuration for the model
│   └── runner_config.json          # Configuration for the batch runner
├── scripts
│   ├── run_batch.sh                # Shell script to execute the batch runner
│   └── cleanup.sh                  # Shell script to clean up results and logs
├── results
│   └── .gitkeep                    # Keeps the results directory tracked by Git
├── logs
│   └── .gitkeep                    # Keeps the logs directory tracked by Git
├── requirements.txt                # Lists project dependencies
├── .gitignore                      # Specifies files to ignore in Git
└── README.md                       # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd sglang-parallel-inference-automation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the model and runner settings in the `config` directory.

## Usage
To run the batch inference process, execute the following command:
```
bash scripts/run_batch.sh
```

This will initiate the inference process 8 times and store the results in the `results` directory, each with a timestamp.

## Result Processing
After running the inference, you can process the results using the `result_processor.py` script. This script provides functions to format and analyze the generated results.

## Cleanup
To clean up old results and logs, run:
```
bash scripts/cleanup.sh
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.