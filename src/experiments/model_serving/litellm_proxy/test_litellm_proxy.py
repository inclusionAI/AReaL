#!/usr/bin/env python3
import os
import sys
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from openai import OpenAI
from loguru import logger

MODEL_NAME = "gpt-4o"
PROXY_URL = "http://localhost:4000"

def test_basic_completion(messages: List[Dict[str, str]], model: str = MODEL_NAME, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Test basic completion using OpenAI SDK
    
    Args:
        messages: List of message dictionaries
        model: Model to use (defaults to MODEL_NAME)
        **kwargs: Additional arguments for completion
        
    Returns:
        Response dictionary or None if error
    """
    try:
        print(f"🧪 Testing completion with model: {model}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print(f"Additional args: {kwargs}")
        print("-" * 50)
        
        # Initialize OpenAI client pointing to the proxy
        client = OpenAI(
            base_url=PROXY_URL,
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key")  # Pass API key to proxy
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        print("✅ Success!")
        print(f"Response: {json.dumps(response.model_dump(), indent=2)}")
        
        # Extract key information
        result = {
            "model": response.model,
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
            "raw_response": response.model_dump()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Completion failed: {e}")
        return None

def test_gpt4o_basic():
    """Test basic conversation with GPT-4o"""
    print("🧪 Test 1: Basic GPT-4o Conversation")
    print("=" * 50)
    
    messages = [
        {"role": "user", "content": "Hello! What LLM are you and what can you do?"}
    ]
    
    return test_basic_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

def test_gpt4o_code_generation():
    """Test code generation with GPT-4o"""
    print("\n🧪 Test 2: GPT-4o Code Generation")
    print("=" * 50)
    
    messages = [
        {"role": "user", "content": "Write a Python function to calculate the factorial of a number with proper error handling."}
    ]
    
    return test_basic_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )

def test_gpt4o_multi_turn():
    """Test multi-turn conversation with GPT-4o"""
    print("\n🧪 Test 3: GPT-4o Multi-turn Conversation")
    print("=" * 50)
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is the population of that city?"}
    ]
    
    return test_basic_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=100
    )

def test_qwen_model():
    """Test the local Qwen model using OpenAI SDK"""
    print("\n🧪 Test 4: Local Qwen Model")
    print("=" * 50)
    
    messages = [
        {"role": "user", "content": "Hello! What model are you?"}
    ]
    
    return test_basic_completion(
        messages=messages,
        model="qwen2.5-7b",  # Use the proxy's model name
        temperature=0.7,
        max_tokens=100
    )

def test_model_listing():
    """Test listing available models"""
    print("\n🧪 Test 5: List Available Models")
    print("=" * 50)
    
    try:
        # Initialize OpenAI client pointing to the proxy
        client = OpenAI(
            base_url=PROXY_URL,
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key")  # Pass API key to proxy
        )
        
        models = client.models.list()
        print("✅ Available models:")
        for model in models.data:
            print(f"  - {model.id}")
        return {"data": [{"id": model.id} for model in models.data]}
            
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return None

def check_environment():
    """Check if required environment variables are set"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OPENAI_API_KEY is set")
        print(f"   Key starts with: {api_key[:8]}...")
    else:
        print("❌ OPENAI_API_KEY is not set")
        print("   Please add it to your .env file: OPENAI_API_KEY=your-key-here")
        return False
    
    return True

def check_proxy_connectivity():
    """Check if the proxy is accessible"""
    print("\n🔍 Proxy Connectivity Check")
    print("=" * 50)
    
    try:
        import requests
        
        response = requests.get(f"{PROXY_URL}/health", timeout=15)
        if response.status_code == 200:
            print("✅ LiteLLM proxy is accessible")
            return True
        else:
            print(f"❌ Proxy responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to proxy: {str(e)}")
        print("   Make sure the proxy is running: ./litellm_proxy.sh start")
        return False

def main():
    """Main test function"""
    print("🚀 OpenAI SDK + LiteLLM Proxy Test Suite")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please set OPENAI_API_KEY.")
        sys.exit(1)
    
    # Check proxy connectivity
    if not check_proxy_connectivity():
        print("\n❌ Proxy connectivity check failed.")
        sys.exit(1)
    
    print("\n✅ All checks passed! Starting tests...")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("GPT-4o Basic", test_gpt4o_basic),
        ("GPT-4o Code Generation", test_gpt4o_code_generation),
        ("GPT-4o Multi-turn", test_gpt4o_multi_turn),
        ("Qwen Local Model", test_qwen_model),
        ("Model Listing", test_model_listing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append(result is not None)
            if result:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! OpenAI SDK is working correctly with LiteLLM proxy.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main()) 