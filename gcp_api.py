#!/usr/bin/env python3
"""
Google Cloud Platform (Gemini) API Connection Test
==================================================

Tests connection to Gemini 2.5 Pro via Google Gen AI SDK.
Validates API key and model access for the compression experiments.

Usage:
    python gcp_api.py

Requirements:
    - GOOGLE_API_KEY environment variable
    - google-genai package installed
"""

import os
import sys
from typing import Optional

def load_api_key() -> Optional[str]:
    """Load Google API key from .env file or environment."""
    # Try to load from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, checking environment variables only")
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found")
        print("   Please either:")
        print("   1. Add GOOGLE_API_KEY=your_key to your .env file, or")
        print("   2. Export as environment variable: export GOOGLE_API_KEY='your-api-key-here'")
        return None
    
    print(f"✅ Google API key loaded: {api_key[:15]}...")
    return api_key

def test_google_genai_import() -> bool:
    """Test if Google Gen AI SDK is installed."""
    try:
        import google.genai as genai
        print("✅ Google Gen AI SDK imported successfully")
        return True
    except ImportError as e:
        print("❌ Error: Google Gen AI SDK not installed")
        print("   Install with: pip install google-genai")
        return False

def test_gemini_connection(api_key: str) -> bool:
    """Test connection to Gemini 2.5 Pro model."""
    try:
        import google.genai as genai
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-pro"
        
        print(f"🔄 Testing connection to {model_name}...")
        
        # Test basic generation
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello, Gemini 2.5 Pro!' to confirm connection"
        )
        
        if response.text and "Hello, Gemini 2.5 Pro" in response.text:
            print(f"✅ {model_name} connection successful!")
            print(f"   Response: {response.text.strip()}")
            return True
        else:
            print(f"❌ {model_name} connection failed: Unexpected response")
            print(f"   Response: {response.text if response.text else 'No response'}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} connection failed: {e}")
        return False

def test_compression_capability(api_key: str) -> bool:
    """Test Gemini's text compression capability."""
    try:
        import google.genai as genai
        
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-pro"
        
        print(f"🔄 Testing compression capability...")
        
        test_text = "The quick brown fox jumps over the lazy dog near the riverbank."
        prompt = f"""Compress this sentence to about 20 characters while preserving meaning. Output ONLY the compressed text:

{test_text}"""
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        if response.text:
            compressed = response.text.strip()
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

def main():
    """Run all Gemini API connection tests."""
    print("🧪 GEMINI 2.5 PRO API CONNECTION TEST")
    print("=" * 50)
    
    # Test 1: Load API key
    api_key = load_api_key()
    if not api_key:
        sys.exit(1)
    
    # Test 2: Check SDK installation
    if not test_google_genai_import():
        sys.exit(1)
    
    # Test 3: Test basic connection
    if not test_gemini_connection(api_key):
        sys.exit(1)
    
    # Test 4: Test compression capability
    if not test_compression_capability(api_key):
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("   Gemini 2.5 Pro is ready for compression experiments")
    print("   You can now run: python extreme_compression_experiment_gemini.py")

if __name__ == "__main__":
    main() 