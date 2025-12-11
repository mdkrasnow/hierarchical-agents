#!/usr/bin/env python3
"""
Comprehensive system test for the no-mock hierarchical agents system.

This script tests core functionality to ensure everything works with real API keys.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.llm import create_llm_client


class SimpleTestOutput(BaseModel):
    """Simple test output format."""
    answer: str
    confidence: float
    reasoning: str


async def test_basic_client_creation():
    """Test 1: Basic client creation"""
    print("🔍 Test 1: Client Creation")
    try:
        client = create_llm_client()
        provider_name = type(client.provider).__name__
        print(f"   ✅ Success: Using {provider_name}")
        return True, provider_name
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False, str(e)


async def test_simple_llm_call():
    """Test 2: Simple LLM call"""
    print("\n🔍 Test 2: Simple LLM Call")
    try:
        client = create_llm_client()
        response = await client.call("What is 2+2? Answer briefly.")
        
        print(f"   Response: {response.content[:50]}...")
        
        # Check if it's a real response (not mock)
        if "Mock LLM response" in response.content:
            print("   ❌ Getting mock response - check API key")
            return False
        else:
            print("   ✅ Success: Real AI response received")
            return True
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_structured_output():
    """Test 3: Structured output"""
    print("\n🔍 Test 3: Structured Output")
    try:
        client = create_llm_client()
        response = await client.call(
            "Answer: What is the capital of France? Include your confidence (0.0-1.0) and reasoning.",
            response_format=SimpleTestOutput
        )
        
        if isinstance(response, SimpleTestOutput):
            print(f"   Answer: {response.answer}")
            print(f"   Confidence: {response.confidence}")
            print("   ✅ Success: Structured output working")
            return True
        else:
            print(f"   ❌ Failed: Got {type(response)} instead of SimpleTestOutput")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_agent_functionality():
    """Test 4: Agent functionality"""
    print("\n🔍 Test 4: Agent Functionality")
    try:
        from critics.single_critic import score_answer
        
        score = await score_answer(
            question="What is photosynthesis?",
            answer="Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
            context="Biology education"
        )
        
        print(f"   Overall score: {score.overall_score}")
        print(f"   Tier: {score.overall_tier}")
        print(f"   Dimensions: {len(score.dimension_scores)}")
        print("   ✅ Success: Agent evaluation working")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def test_performance():
    """Test 5: Performance metrics"""
    print("\n🔍 Test 5: Performance Check")
    try:
        client = create_llm_client()
        
        start_time = time.time()
        response = await client.call("Quick test - what is 1+1?")
        latency = time.time() - start_time
        
        print(f"   Latency: {latency:.2f}s")
        
        if hasattr(response, 'latency_ms') and response.latency_ms:
            print(f"   Reported latency: {response.latency_ms:.0f}ms")
        
        if hasattr(response, 'token_usage') and response.token_usage:
            print(f"   Token usage: {response.token_usage}")
        
        print("   ✅ Success: Performance metrics available")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


async def main():
    """Run comprehensive system test."""
    print("🚀 Hierarchical Agents System Test")
    print("=" * 50)
    
    # Check environment first
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No API key detected!")
        print("   Set GEMINI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("   Get Gemini key: https://aistudio.google.com/apikey")
        print("\n   Example: export GEMINI_API_KEY=your_key_here")
        return False
    
    print("✅ API key detected")
    
    # Run tests
    tests = [
        ("Client Creation", test_basic_client_creation),
        ("Simple LLM Call", test_simple_llm_call),
        ("Structured Output", test_structured_output),
        ("Agent Functionality", test_agent_functionality),
        ("Performance Check", test_performance)
    ]
    
    results = []
    provider_info = None
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if isinstance(result, tuple):
                success, info = result
                if test_name == "Client Creation":
                    provider_info = info
            else:
                success = result
            results.append(success)
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if provider_info and "Provider" in str(provider_info):
        print(f"Provider: {provider_info}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is working correctly with real AI APIs")
        print("\nReady for production use! 🚀")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("Check the error messages above and your API key setup")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)