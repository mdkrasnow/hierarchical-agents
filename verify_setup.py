#!/usr/bin/env python3
"""
Quick setup verification for the hierarchical agents system.

Run this first to check if your environment is ready for testing.
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly set up."""
    print("🔍 Environment Check")
    print("=" * 30)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    if sys.version_info < (3, 9):
        print("⚠️  Python 3.9+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"\nAPI Keys:")
    if gemini_key:
        if gemini_key.startswith("AIza") and len(gemini_key) > 30:
            print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}...")
        else:
            print(f"⚠️  GEMINI_API_KEY looks invalid: {gemini_key[:10]}...")
    else:
        print("❌ GEMINI_API_KEY: Not set")
    
    if claude_key:
        print(f"✅ ANTHROPIC_API_KEY: {claude_key[:10]}...")
    else:
        print("❌ ANTHROPIC_API_KEY: Not set")
    
    if not gemini_key and not claude_key:
        print("\n❌ No API keys found!")
        print("You need at least one API key to test the system.")
        print("\nQuick setup:")
        print("1. Get Gemini API key: https://aistudio.google.com/apikey")
        print("2. Run: export GEMINI_API_KEY=your_key_here")
        print("3. Or add to .env file: echo 'GEMINI_API_KEY=your_key' >> .env")
        return False
    
    # Check imports
    print(f"\nDependency Check:")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from utils.llm import create_llm_client
        print("✅ Core modules importable")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    try:
        import google.genai
        print("✅ Google GenAI available")
    except ImportError:
        print("❌ Google GenAI missing")
        print("Run: pip install google-genai")
        return False
    
    # Check client creation
    print(f"\nClient Test:")
    try:
        client = create_llm_client()
        provider_type = type(client.provider).__name__
        print(f"✅ LLM client created: {provider_type}")
        return True
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False

def main():
    print("🚀 Hierarchical Agents Setup Verification")
    print("=" * 50)
    
    if check_environment():
        print("\n" + "=" * 50)
        print("🎉 Setup verification PASSED!")
        print("\nYour system is ready for testing.")
        print("\nNext steps:")
        print("1. Run: python test_system.py")
        print("2. Try: python scripts/score_single_answer.py interactive")
        print("3. Test agents: python scripts/run_eval_agent.py")
    else:
        print("\n" + "=" * 50)
        print("❌ Setup verification FAILED!")
        print("\nPlease fix the issues above before testing.")
    
    return check_environment()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)