# External Providers

Test scripts for external model providers.

## Setup

Set environment variables in `.env`:
- `TOGETHER_API_KEY` - Together AI API key
- `BASETEN_API_KEY` - Baseten API key  
- `BASETEN_MODEL_URL` - Baseten model endpoint URL

## Usage

```bash
# Test Together AI
python test_together.py

# Test Baseten
python test_baseten.py
``` 