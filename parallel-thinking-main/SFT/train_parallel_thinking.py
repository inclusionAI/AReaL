import os
import sys
import json
import subprocess
from pathlib import Path
import yaml

def setup_environment():
    """Setup environment variables and paths"""
    os.environ['BNB_CUDA_VERSION'] = '125'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,5'  # Adjust based on your GPU setup
def validate_data():
    """Validate the prepared training data"""
    print("Validating training data...")
    
    with open('data/parallel_thinking_train.jsonl', 'r', encoding='utf-8') as f:
        sample_count = 0
        for line in f:
            if sample_count >= 3:  # Check first 3 samples
                break
            
            data = json.loads(line.strip())
            conversations = data.get('conversations', [])
            
            if len(conversations) == 2:
                human_msg = conversations[0]['value']
                gpt_msg = conversations[1]['value']
                
                print(f"\n--- Sample {sample_count + 1} ---")
                print(f"Human: {human_msg[:100]}...")
                print(f"GPT: {gpt_msg[:200]}...")
                print(f"Has reasoning process: {'<reasoning_process>' in gpt_msg}")
                
            sample_count += 1
def prepare_data():
    """Convert and prepare training data"""
    print("Converting data to ShareGPT format...")
    
    converted_data = []
    
    with open('/home/zhangzy/parallel-thinking/data_processing/output_submit_all_1000.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get('original_problem', '')
            main_thread = data.get('main_thread', '')
            
            if question and main_thread:
                # Extract just the assistant's response part
                if "Assistant: " in main_thread:
                    # Split and get the assistant's response
                    assistant_response = main_thread.split("Assistant: ", 1)[1].strip()
                    
                    # Ensure we have the reasoning process
                    if "<reasoning_process>" in assistant_response:
                        conversation = {
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": question  # Fixed: was "Question"
                                },
                                {
                                    "from": "gpt", 
                                    "value": assistant_response
                                }
                            ]
                        }
                        converted_data.append(conversation)
                    else:
                        print(f"Warning: No reasoning process found in entry {data.get('problem_index', 'unknown')}")
    
    print(f"Prepared {len(converted_data)} training examples")
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    with open('data/parallel_thinking_train.jsonl', 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create dataset info
    dataset_info = {
        "parallel_thinking": {
            "file_name": "parallel_thinking_train.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value", 
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    }
    
    with open('data/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

def find_model_path():
    """Find the correct model path, checking for snapshots directory"""
    base_path = "/home/zhangzy/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
    
    # Check if snapshots directory exists
    snapshots_path = Path(base_path) / "snapshots"
    if snapshots_path.exists():
        snapshots = list(snapshots_path.iterdir())
        if snapshots:
            model_path = str(snapshots[0])
            print(f"Found model in snapshots: {model_path}")
            
            # Verify config.json exists
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                return model_path
            else:
                print(f"Warning: config.json not found in {model_path}")
    
    # Check base path
    config_path = Path(base_path) / "config.json"
    if config_path.exists():
        return base_path
    
    # If local model not found or incomplete, use HuggingFace identifier
    print("Local model not found or incomplete, using HuggingFace model identifier")
    return "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
def create_deepspeed_config():
    """Create DeepSpeed configuration for full fine-tuning"""
    deepspeed_config = {
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,  # ZeRO Stage 2
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": 6,  # Total across all GPUs
        "train_micro_batch_size_per_gpu": 2,
        "wall_clock_breakdown": False
    }
    
    with open('deepspeed_config.json', 'w') as f:
        json.dump(deepspeed_config, f, indent=2)

def create_monitoring_script():
    """Create a script to monitor model outputs during training"""
    monitoring_code = '''
import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import glob
import os

class TrainingMonitor:
    def __init__(self, base_model_path, output_dir, test_samples=5):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.test_samples = test_samples
        self.test_questions = []
        self.load_test_questions()
        
    def load_test_questions(self):
        """Load test questions from training data"""
        with open('data/parallel_thinking_train.jsonl', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.test_samples:
                    break
                data = json.loads(line.strip())
                question = data['conversations'][0]['value']
                expected = data['conversations'][1]['value']
                self.test_questions.append({
                    'question': question,
                    'expected': expected[:200] + '...',  # Truncate for display
                    'index': i
                })
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint"""
        if not os.path.exists(self.output_dir):
            return None
            
        checkpoints = glob.glob(f"{self.output_dir}/checkpoint-*")
        if not checkpoints:
            return None
            
        # Sort by checkpoint number
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        return checkpoints[-1]
    
    def test_checkpoint(self, checkpoint_path):
        """Test a specific checkpoint"""
        print(f"\\n{'='*80}")
        print(f"Testing checkpoint: {checkpoint_path}")
        print(f"{'='*80}")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            model.eval()
            
            # Test each question
            for test_case in self.test_questions:
                print(f"\\n--- Problem {test_case['index'] + 1} ---")
                print(f"Question: {test_case['question'][:100]}...")
                
                # Format prompt
                prompt = f"Human: {test_case['question']}\\n\\nAssistant:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                model_output = response.split("Assistant:")[-1].strip()
                
                print(f"Model Output: {model_output[:200]}...")
                print(f"Expected: {test_case['expected']}")
                print(f"Has <reasoning_process>: {'<reasoning_process>' in model_output}")
                print("-" * 50)
                
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing checkpoint {checkpoint_path}: {e}")
    
    def monitor_training(self, check_interval=300):  # Check every 5 minutes
        """Continuously monitor training and test new checkpoints"""
        tested_checkpoints = set()
        
        print(f"Starting training monitor. Checking every {check_interval} seconds...")
        print(f"Monitoring directory: {self.output_dir}")
        
        while True:
            try:
                latest_checkpoint = self.find_latest_checkpoint()
                
                if latest_checkpoint and latest_checkpoint not in tested_checkpoints:
                    self.test_checkpoint(latest_checkpoint)
                    tested_checkpoints.add(latest_checkpoint)
                else:
                    print(f"\\rWaiting for new checkpoint... Latest: {latest_checkpoint or 'None'}", end='')
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\\nMonitoring stopped by user.")
                break
            except Exception as e:
                print(f"\\nError in monitoring: {e}")
                time.sleep(check_interval)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python monitor_training.py <base_model_path> <output_dir>")
        sys.exit(1)
    
    base_model_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    monitor = TrainingMonitor(base_model_path, output_dir, test_samples=3)
    monitor.monitor_training(check_interval=180)  # Check every 3 minutes
'''
    
    with open('monitor_training.py', 'w') as f:
        f.write(monitoring_code)
    
    print("Created monitor_training.py")

def create_training_config():
    """Create training configuration file without evaluation"""
    model_path = find_model_path()
    time = subprocess.run(['date', '+%Y%m%d_%H%M%S'], capture_output=True, text=True).stdout.strip()
    config = {
        # Model settings
        'model_name_or_path': model_path,
        'stage': 'sft',
        'do_train': True,
        'deepspeed': 'deepspeed_config.json',
        'finetuning_type': 'full',
        
        # Dataset settings
        'dataset': 'parallel_thinking',
        'template': 'qwen',
        'cutoff_len': 2048,
        'train_on_prompt': False,
        'mask_history': True,
        
        # Training settings - MADE MORE CONSERVATIVE
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 2,
        'learning_rate': 1e-5,  # Conservative learning rate
        'num_train_epochs': 3,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'max_grad_norm': 0.5,
        'bf16': True,
        'gradient_checkpointing': True,
        
        # Output settings - NO EVALUATION
        'output_dir': f'/nvme0n1/zzy_model/saves_{time}/deepseek-parallel-thinking',
        'logging_steps': 5,  # Log every 5 steps
        'save_steps': 50,   # Save every 50 steps
        'save_total_limit': 10,  # Keep more checkpoints for monitoring
        'save_strategy': 'steps',
        # REMOVED: 'eval_strategy': 'steps',
        # REMOVED: 'eval_steps': 50,
        
        # Generation settings
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'max_new_tokens': 512,
    }
    
    with open('train_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Training config created with model path: {model_path}")
    return f'/nvme0n1/zzy_model/saves_{time}/deepseek-parallel-thinking'

def create_deepspeed_config():
    """Create DeepSpeed configuration with better stability"""
    deepspeed_config = {
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": 2,  # Match config
        "gradient_clipping": 0.5,  # Lower gradient clipping
        "steps_per_print": 5,
        "train_batch_size": 6,  # 3 GPUs * 1 batch * 2 accumulation
        "train_micro_batch_size_per_gpu": 1,  # Match config
        "wall_clock_breakdown": False
    }
    
    with open('deepspeed_config.json', 'w') as f:
        json.dump(deepspeed_config, f, indent=2)

def run_training():
    """Start the training process"""
    print("Starting training...")
    
    # Ensure LLaMA-Factory is available
    if not os.path.exists('LLaMA-Factory'):
        print("Cloning LLaMA-Factory...")
        subprocess.run(['git', 'clone', 'https://github.com/hiyouga/LLaMA-Factory.git'])
        subprocess.run(['pip', 'install', '-e', '.[torch,metrics]'], cwd='LLaMA-Factory')
    
    # Copy data to LLaMA-Factory directory
    subprocess.run(['cp', '-r', 'data/', 'LLaMA-Factory/'])
    subprocess.run(['cp', 'train_config.yaml', 'LLaMA-Factory/'])
    subprocess.run(['cp', 'deepspeed_config.json', 'LLaMA-Factory/'])
    subprocess.run(['cp', 'monitor_training.py', 'LLaMA-Factory/'])  # Copy monitoring script
    
    # Verify files exist in LLaMA-Factory directory
    required_files = ['train_config.yaml', 'deepspeed_config.json', 'data/parallel_thinking_train.jsonl']
    for file in required_files:
        file_path = f'LLaMA-Factory/{file}'
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found!")
            return False
        else:
            print(f"✓ Found: {file_path}")
    
    # Start training in background and monitoring
    try:
        print("Starting training process...")
        cmd = ['llamafactory-cli', 'train', 'train_config.yaml']
        print(f"Running command: {' '.join(cmd)}")
        
        # Start training process in background
        import threading
        
        def run_training_process():
            subprocess.run(cmd, cwd='LLaMA-Factory')
        
        training_thread = threading.Thread(target=run_training_process)
        training_thread.start()
        
        # Give training time to start
        import time
        time.sleep(30)
        
        # Start monitoring
        print("\\nStarting training monitor...")
        print("Press Ctrl+C to stop monitoring (training will continue)")
        
        # Get output directory from config
        with open('train_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            output_dir = config['output_dir']
        
        model_path = find_model_path()
        monitor_cmd = ['python', 'monitor_training.py', model_path, output_dir]
        subprocess.run(monitor_cmd, cwd='LLaMA-Factory')
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\\nMonitoring stopped. Training continues in background.")
        return True

def test_model():
    """Test the fine-tuned model"""
    print("Testing the fine-tuned model...")
    
    # Updated test script for DeepSeek model
    model_path = find_model_path()
    
    test_code = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

# Model path handling
model_path = "{model_path}"

# If using HuggingFace identifier, load directly
if not model_path.startswith("/"):
    print("Loading model from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
else:
    # Local model path
    print(f"Loading model from local path: {{model_path}}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

# Load fine-tuned adapter
try:
    model = PeftModel.from_pretrained(base_model, "saves/deepseek-parallel-thinking")
    print("Fine-tuned adapter loaded successfully!")
except Exception as e:
    print(f"Error loading adapter: {{e}}")
    print("Using base model only...")
    model = base_model

# Test with a sample question
test_question = "In triangle ABC with incenter I, what is the relationship between the angle bisectors?"

# Format prompt for DeepSeek/Qwen format
prompt = test_question  # Just the question, no special formatting
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Test Question:", test_question)
print("Model Response:", response.split("Assistant:")[-1].strip())
'''
    
    with open('test_model.py', 'w') as f:
        f.write(test_code)
    
    print("Test script created as 'test_model.py'")
    print("Run: python test_model.py to test your fine-tuned model")

def main():
    """Main training pipeline"""
    print("=== LLaMA Factory Fine-tuning Pipeline with Monitoring ===")
    
    # Setup
    setup_environment()
    
    # Prepare data
    prepare_data()
    validate_data()
    
    # Create monitoring script
    create_monitoring_script()
    
    # Create config
    output_dir = create_training_config()
    create_deepspeed_config()
    
    # Run training with monitoring
    success = run_training()
    
    # Create test script
    test_model()
    
    if success:
        print("\\n=== Training Started with Monitoring! ===")
        print(f"Training output will be saved to: {output_dir}")
        print("Monitor script will test each checkpoint automatically.")
        print("\\nTo manually test a checkpoint later:")
        print(f"python monitor_training.py {find_model_path()} {output_dir}")
    else:
        print("\\n=== Training Failed! ===")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()
