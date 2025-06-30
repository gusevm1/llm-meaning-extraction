#!/usr/bin/env python3
"""
Setup script for LLM Meaning Extraction project.
Helps users get started with the project setup.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_env_file():
    """Set up environment file for API keys."""
    env_file = Path('.env')
    template_file = Path('env_template.txt')
    
    if env_file.exists():
        print("⚠️  .env file already exists. Skipping creation.")
        return True
    
    if not template_file.exists():
        print("❌ Template file not found!")
        return False
    
    try:
        # Copy template to .env
        with open(template_file, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from template")
        print("📝 Please edit .env file and add your API keys!")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_api_keys():
    """Check if API keys are configured."""
    env_file = Path('.env')
    if not env_file.exists():
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Check if template values are still present
    if 'your_openai_api_key_here' in content:
        return False
    
    return True

def test_openai_setup():
    """Test OpenAI setup if API key is configured."""
    if not check_api_keys():
        print("⚠️  Please configure your API keys in .env file before testing")
        return False
    
    print("🧪 Testing OpenAI API setup...")
    try:
        from openai_api import OpenAIHandler
        handler = OpenAIHandler()
        if handler.test_connection():
            print("✅ OpenAI API test passed!")
            return True
        else:
            print("❌ OpenAI API test failed!")
            return False
    except Exception as e:
        print(f"❌ OpenAI test error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 LLM Meaning Extraction Project Setup")
    print("=" * 40)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        return
    
    # Step 2: Set up environment file
    if not setup_env_file():
        return
    
    # Step 3: Check if API keys are configured
    if check_api_keys():
        print("✅ API keys appear to be configured")
        
        # Step 4: Test OpenAI setup
        if test_openai_setup():
            print("\n🎉 Setup complete! You can now run:")
            print("   python openai_api.py")
        else:
            print("\n⚠️  Setup complete but API test failed.")
            print("   Please check your API keys in .env file")
    else:
        print("\n📝 Next steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run: python openai_api.py")
    
    print("\n📚 See README.md for detailed instructions")

if __name__ == "__main__":
    main() 