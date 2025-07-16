import ollama
import subprocess

class OllamaManager:
    def __init__(self):
        self.process = None
    
    def start_server(self):
        """Start Ollama server if not running"""
        try:
            # Check if already running
            ollama.list()
            print("Ollama server already running")
        except:
            print("Starting Ollama server...")
            self.process = subprocess.Popen(['ollama', 'serve'])
            import time
            time.sleep(5)  # Wait for startup
    
    def pull_model(self, model_name):
        """Download a model if not exists"""
        try:
            ollama.pull(model_name)
            print(f"Model {model_name} ready")
        except Exception as e:
            print(f"Error pulling model: {e}")

    def get_model_list(self):
        """Get the list of available models"""
        return ollama.list()
        
    def chat(self, model, message):
        """Send a chat message"""
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': message}]
        )
        return response['message']['content']
    
    def stop_server(self):
        """Stop the Ollama server"""
        if self.process:
            self.process.terminate()
            print("Ollama server stopped")
