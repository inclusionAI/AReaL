#!/usr/bin/env python3
"""
Simple example demonstrating the auto-server-launch feature.
This script shows how to use the MultiverseGeneratorNew class with automatic server management.
"""

from generation import MultiverseGeneratorNew


def main():
    # Example problem from AIME
    problem = r"""A list of positive integers has the following properties:
$\bullet$ The sum of the items in the list is $30$.
$\bullet$ The unique mode of the list is $9$.
$\bullet$ The median of the list is a positive integer that does not appear in the list itself.
Find the sum of the squares of all the items in the list."""
    
    print("="*70)
    print("Example: Multiverse Generation with Auto Server Launch")
    print("="*70)
    print("\nThis example demonstrates:")
    print("1. Automatic SGLang server launch")
    print("2. Generation with parallel path detection")
    print("3. Automatic server shutdown")
    print("="*70)
    
    # Set your model path here
    model_path = "Qwen/Qwen3-8B"  # Change this to your model path
    
    print(f"\nModel: {model_path}")
    print("Port: 30000")
    print("TP Size: 8")
    
    input("\nPress Enter to continue...")
    
    try:
        # Initialize generator - this will launch the server automatically
        print("\n" + "="*70)
        print("Initializing generator and launching server...")
        print("="*70)
        
        generator = MultiverseGeneratorNew(
            model_path=model_path,
            host="127.0.0.1",
            port=30000,
            tp_size=8,
            launch_server=True  # <-- Key parameter: auto-launch server
        )
        
        print("\n✅ Server launched and ready!")
        
        # Run generation
        print("\n" + "="*70)
        print("Running generation...")
        print("="*70)
        
        result = generator.run_generation(
            problem=problem,
            max_normal_tokens=10000,
            max_path_tokens=10000,
            max_conclusion_tokens=10000,
            max_total_tokens=32768,
            temperature=0.7
        )
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        generator.print_result(result)
        
        # Server will automatically shut down when this script exits
        print("\n" + "="*70)
        print("Server will shut down automatically when exiting...")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Server will shut down.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done! Server has been shut down.")


if __name__ == "__main__":
    main()
