
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
        print(f"\n{'='*80}")
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
                print(f"\n--- Problem {test_case['index'] + 1} ---")
                print(f"Question: {test_case['question'][:100]}...")
                
                # Format prompt
                prompt = f"Human: {test_case['question']}\n\nAssistant:"
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
                    print(f"\rWaiting for new checkpoint... Latest: {latest_checkpoint or 'None'}", end='')
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user.")
                break
            except Exception as e:
                print(f"\nError in monitoring: {e}")
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
