#!/usr/bin/env python3

import argparse
import asyncio
import json
import aiohttp

async def test_serp_search_tool(url: str):
    """Test the SERP search tool with sample queries."""
    
    # Test data with different search query formats
    test_data = {
        "trajectory_ids": ["serp_test_1", "serp_test_2", "serp_test_3", "serp_test_4"],
        "actions": [
            "<search>artificial intelligence latest news</search>",
            "```search\nPython pandas tutorial\n```",
            "What is machine learning? <search>machine learning basics</search>",
            "<search>climate change 2024</search>"
        ],
        "extra_fields": [
            {"is_last_step": False},
            {"is_last_step": False},
            {"is_last_step": False},
            {"is_last_step": True}  # This will clean up the environment
        ]
    }

    async with aiohttp.ClientSession() as session:
        print(f"Testing SERP Search Tool at {url}")
        print("=" * 50)
        
        # Send the request
        async with session.post(url, json=test_data) as response:
            if response.status == 200:
                result = await response.json()
                
                print("Request successful!")
                print(f"Status: {response.status}")
                print("\nResults:")
                
                observations = result.get("observations", [])
                dones = result.get("dones", [])
                valids = result.get("valids", [])
                
                for i, (obs, done, valid) in enumerate(zip(observations, dones, valids)):
                    print(f"\n--- Test {i+1} ---")
                    print(f"Query: {test_data['actions'][i]}")
                    print(f"Valid: {valid}")
                    print(f"Done: {done}")
                    print(f"Observation (first 800 chars): {obs[:800]}...")
                    if len(obs) > 800:
                        print(f"[Truncated - Total length: {len(obs)} characters]")
                    print("-" * 40)
                
            else:
                print(f"Request failed with status: {response.status}")
                error_text = await response.text()
                print(f"Error: {error_text}")

def main():
    parser = argparse.ArgumentParser(description="Test SERP Search Tool")
    parser.add_argument("tool_name", help="Tool name (should be 'serp_search')")
    parser.add_argument("--url", default="http://localhost:5500/get_observation", 
                       help="URL of the tool server endpoint")
    
    args = parser.parse_args()
    
    if args.tool_name != "serp_search":
        print(f"Warning: Expected tool name 'serp_search', got '{args.tool_name}'")
    
    print(f"Testing SERP Search Tool")
    print(f"Server URL: {args.url}")
    print("Note: Make sure you have set SERP_API_KEY environment variable")
    print("or configured the tool server with SerpAPI credentials.")
    print("You can get a free API key from https://serpapi.com/")
    print()
    
    # Run the async test
    asyncio.run(test_serp_search_tool(args.url))

if __name__ == "__main__":
    main() 