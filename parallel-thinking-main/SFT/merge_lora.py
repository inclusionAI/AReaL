import subprocess
import os
import yaml
from pathlib import Path

def export_merged_model_llamafactory_offline():
    """Export with local model path - no internet required"""
    
    # Ensure LLaMA-Factory directory exists
    if not os.path.exists('LLaMA-Factory'):
        print("Error: LLaMA-Factory directory not found!")
        return None
    
    # Use the local model path from your cache
    local_model_path = "/home/zhangzy/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    
    # Check if local model exists
    if not os.path.exists(local_model_path):
        print(f"Error: Local model not found at {local_model_path}")
        return None
    
    # Find adapter path
    adapter_path = verify_files_exist()
    if not adapter_path:
        return None
    
    time = subprocess.run(['date', '+%Y%m%d_%H%M%S'], capture_output=True, text=True).stdout.strip()
    
    # Create export config with local paths only
    export_config = {
        'model_name_or_path': local_model_path,  # Use local path
        'adapter_name_or_path': adapter_path,
        'template': 'qwen',
        'finetuning_type': 'lora',
        'export_dir': f'/nvme0n1/zzy_model/exported_models/deepseek-parallel-thinking-merged_{time}',
        'export_size': 2,
        'export_device': 'cuda',
        'export_legacy_format': False
    }
    
    # Create output directory
    os.makedirs('exported_model', exist_ok=True)
    
    # Create export config file in LLaMA-Factory directory
    config_path = Path('LLaMA-Factory/export_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(export_config, f, default_flow_style=False)
    
    print(f"Created export config at: {config_path}")
    print("Export configuration:")
    for key, value in export_config.items():
        print(f"  {key}: {value}")
    
    # Set environment variables to force offline mode
    env = os.environ.copy()
    env['HF_HUB_OFFLINE'] = '1'  # Force offline mode
    env['TRANSFORMERS_OFFLINE'] = '1'  # Force transformers offline
    
    # Run export command
    try:
        cmd = ['llamafactory-cli', 'export', 'export_config.yaml']
        print(f"Running command: {' '.join(cmd)} (offline mode)")
        
        result = subprocess.run(
            cmd, 
            cwd='LLaMA-Factory', 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env  # Use offline environment
        )
        
        print("Model export completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        
        return export_config['export_dir']
        
    except subprocess.TimeoutExpired:
        print("Export timed out after 30 minutes")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Export failed with return code: {e.returncode}")
        if e.stderr:
            print("Error output:", e.stderr)
        if e.stdout:
            print("Standard output:", e.stdout)
        return None

def verify_files_exist():
    """Verify that required files exist before export"""
    
    # Try to find the actual adapter path
    possible_paths = [
        'saves_20250621_210441/deepseek-parallel-thinking',  # Your specific path
        'saves/deepseek-parallel-thinking',
        'saves_*/deepseek-parallel-thinking'
    ]
    
    for path_pattern in possible_paths:
        if '*' in path_pattern:
            import glob
            full_pattern = f'LLaMA-Factory/{path_pattern}'
            matches = glob.glob(full_pattern)
            if matches:
                relative_path = matches[0].replace('LLaMA-Factory/', '')
                print(f"Found adapter at: {relative_path}")
                return relative_path
        else:
            full_path = f'LLaMA-Factory/{path_pattern}'
            if os.path.exists(full_path):
                print(f"Found adapter at: {path_pattern}")
                return path_pattern
    
    print("Available directories in LLaMA-Factory:")
    llamafactory_dir = Path('LLaMA-Factory')
    if llamafactory_dir.exists():
        for item in llamafactory_dir.rglob('*deepseek*'):
            if item.is_dir():
                relative_path = item.relative_to(llamafactory_dir)
                print(f"  - {relative_path}")
    
    return None

if __name__ == "__main__":
    result = export_merged_model_llamafactory_offline()
    if result:
        print(f"\n=== Export completed! ===")
        print(f"Merged model saved at: {result}")
    else:
        print("\n=== Export failed! ===")