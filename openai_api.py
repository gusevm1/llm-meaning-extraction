#!/usr/bin/env python3
"""
OpenAI API Connection Test
==========================

Tests connection to GPT-4.1 (gpt-4o-2024-11-20) for compression experiments.
Validates API key and model access.

Usage:
    python openai_api.py

Requirements:
    - OPENAI_API_KEY environment variable
    - openai package installed
"""

import os
import sys
from typing import Optional

def load_api_key() -> Optional[str]:
    """Load OpenAI API key from .env file or environment."""
    # Try to load from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, checking environment variables only")
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        print("   Please either:")
        print("   1. Add OPENAI_API_KEY=your_key to your .env file, or")
        print("   2. Export as environment variable: export OPENAI_API_KEY='your-api-key-here'")
        return None
    
    print(f"✅ OpenAI API key loaded: {api_key[:15]}...")
    return api_key

def test_openai_import() -> bool:
    """Test if OpenAI SDK is installed."""
    try:
        import openai
        print(f"✅ OpenAI SDK imported successfully (version: {openai.__version__})")
        return True
    except ImportError:
        print("❌ Error: OpenAI SDK not installed")
        print("   Install with: pip install openai")
        return False

def test_gpt4_connection(api_key: str) -> bool:
    """Test connection to GPT-4.1 model."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o-2024-11-20"  # GPT-4.1
        
        print(f"🔄 Testing connection to {model_name} (GPT-4.1)...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, GPT-4.1!' to confirm connection"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            if "Hello, GPT-4.1" in content:
                print(f"✅ {model_name} connection successful!")
                print(f"   Response: {content}")
                return True
            else:
                print(f"❌ {model_name} connection failed: Unexpected response")
                print(f"   Response: {content}")
                return False
        else:
            print(f"❌ {model_name} connection failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} connection failed: {e}")
        return False

def test_compression_capability(api_key: str) -> bool:
    """Test GPT-4.1's text compression capability."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o-2024-11-20"
        
        print(f"🔄 Testing compression capability...")
        
        test_text = "The quick brown fox jumps over the lazy dog near the riverbank."
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a text compression expert. Output ONLY the compressed text, no explanations."},
                {"role": "user", "content": f"Compress this sentence to about 20 characters while preserving meaning:\n\n{test_text}"}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        if response.choices and response.choices[0].message.content:
            compressed = response.choices[0].message.content.strip()
            compression_ratio = len(compressed) / len(test_text)
            
            print(f"✅ Compression test successful!")
            print(f"   Original: {test_text} ({len(test_text)} chars)")
            print(f"   Compressed: {compressed} ({len(compressed)} chars)")
            print(f"   Ratio: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
            return True
        else:
            print(f"❌ Compression test failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ Compression test failed: {e}")
        return False

def test_temperature_support(api_key: str) -> bool:
    """Test that GPT-4.1 supports temperature parameter (unlike o3)."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o-2024-11-20"
        
        print(f"🔄 Testing temperature parameter support...")
        
        # Test with temperature 0.7
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": "Say 'Temperature works!' in a creative way"}
            ],
            max_tokens=30,
            temperature=0.7
        )
        
        if response.choices and response.choices[0].message.content:
            print(f"✅ Temperature parameter supported!")
            print(f"   Creative response: {response.choices[0].message.content.strip()}")
            return True
        else:
            print(f"❌ Temperature test failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ Temperature test failed: {e}")
        return False

def main():
    """Run all OpenAI API connection tests."""
    print("🧪 GPT-4.1 API CONNECTION TEST")
    print("=" * 40)
    
    # Test 1: Load API key
    api_key = load_api_key()
    if not api_key:
        sys.exit(1)
    
    # Test 2: Check SDK installation
    if not test_openai_import():
        sys.exit(1)
    
    # Test 3: Test basic connection
    if not test_gpt4_connection(api_key):
        sys.exit(1)
    
    # Test 4: Test compression capability
    if not test_compression_capability(api_key):
        sys.exit(1)
    
    # Test 5: Test temperature support
    if not test_temperature_support(api_key):
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("   GPT-4.1 is ready for compression experiments")
    print("   You can now run: python extreme_compression_experiment_openai.py")

if __name__ == "__main__":
    main() 