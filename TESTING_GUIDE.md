# Testing Guide - Production System

Since all mock functionality has been removed, testing requires real API keys. Here are comprehensive testing approaches for your system.

## 🔑 Prerequisites

### Get API Key
1. **Gemini (Recommended)**: https://aistudio.google.com/apikey

### Set Environment Variable
```bash
# Option 1: Export for session
export GEMINI_API_KEY=your_api_key_here

# Option 2: Add to .env file  
echo "GEMINI_API_KEY=your_api_key_here" >> .env

# Option 3: One-time use
GEMINI_API_KEY=your_key python script.py
```

## 🧪 Testing Levels

### 1. **Quick Smoke Tests**

**Test Basic LLM Functionality:**
```bash
# Test that client creation works
python -c "from src.utils.llm import create_llm_client; print('✅ Client created:', type(create_llm_client().provider).__name__)"
```

**Test Simple Agent Call:**
```bash
# Run basic evaluation agent
python scripts/run_eval_agent.py
```

### 2. **Component Testing**


**Test Multi-Critic Debate:**
```bash
# Interactive debate system
python scripts/score_with_debate.py interactive

# Quick test with command line
python scripts/score_with_debate.py quick "What causes rain?" "Rain happens when water evaporates, forms clouds, and falls back down."
```

### 3. **Agent System Testing**

**Test Teacher Analysis:**
```bash
python scripts/run_teacher_agent.py --demo
```

**Test School-Level Analysis:**
```bash
python scripts/run_school_agent.py --demo
```

**Test District-Level Analysis:**
```bash
python scripts/run_district_agent.py --demo
```

### 4. **Advanced Testing**

**Test with Different Providers:**
```bash
# Force Gemini
python scripts/score_single_answer.py interactive --provider gemini

# Force Claude (if you have ANTHROPIC_API_KEY)
python scripts/score_single_answer.py interactive --provider claude
```

**Test Batch Processing:**
```bash
python scripts/run_eval_agent.py  # Includes batch tests
```

## 📋 Test Scenarios

### Scenario 1: Content Evaluation
```bash
# Test question-answer evaluation
python scripts/score_single_answer.py quick \
  "Explain climate change" \
  "Climate change is the warming of Earth due to greenhouse gases from human activities like burning fossil fuels."
```

### Scenario 2: Educational Assessment  
```bash
# Test teacher evaluation system
python scripts/run_teacher_agent.py --demo
```

### Scenario 3: Multi-Perspective Analysis
```bash
# Test critic debate system
python scripts/score_with_debate.py quick \
  "What is artificial intelligence?" \
  "AI is computer systems that can perform tasks that typically require human intelligence, like learning and decision-making."
```

## 🔧 Debugging Failed Tests

### Common Issues & Solutions

**❌ "No LLM API key found"**
```bash
# Check your environment
echo $GEMINI_API_KEY
# If empty, set it:
export GEMINI_API_KEY=your_actual_key
```

**❌ "API key required for Gemini provider"**
```bash
# Your key might be invalid or expired
# Get a new one from: https://aistudio.google.com/apikey
```

**❌ Rate Limit Errors**
```bash
# Gemini has rate limits on free tier
# Wait a few minutes between tests, or upgrade to paid tier
```

**❌ Import Errors**
```bash
# Install dependencies
pip install -r requirements.txt

# Or just Gemini package
pip install google-genai
```

## 🚀 Automated Testing

### Create a Test Suite
```bash
#!/bin/bash
# test_system.sh

echo "🔍 Testing System Functionality"

# Test 1: Basic client creation
echo "Test 1: Client Creation"
python -c "from src.utils.llm import create_llm_client; create_llm_client(); print('✅ Pass')" || echo "❌ Fail"

# Test 2: Simple evaluation  
echo "Test 2: Simple Evaluation"
echo "What is 2+2?" | python scripts/score_single_answer.py quick "2+2 equals 4" || echo "❌ Fail"

# Test 3: Agent system
echo "Test 3: Agent System"  
timeout 30 python scripts/run_eval_agent.py && echo "✅ Pass" || echo "❌ Fail"

echo "Testing complete!"
```

### Performance Testing
```python
# performance_test.py
import asyncio
import time
from src.utils.llm import create_llm_client

async def test_performance():
    client = create_llm_client()
    
    # Test single call performance
    start = time.time()
    response = await client.call("What is 1+1?")
    latency = time.time() - start
    print(f"Single call latency: {latency:.2f}s")
    
    # Test batch performance
    start = time.time()
    batch_requests = [("What is 2+2?", None), ("What is 3+3?", None)]
    batch_result = await client.call_batch(batch_requests)
    batch_latency = time.time() - start
    print(f"Batch call latency: {batch_latency:.2f}s")
    print(f"Successful: {batch_result.successful_count}")

if __name__ == "__main__":
    asyncio.run(test_performance())
```

## 📊 Expected Results

### ✅ Successful Test Indicators:
- **Provider Type**: `GeminiLLMProvider` (not Mock)
- **Real Responses**: Contextual, varied answers
- **Token Usage**: Actual token counts in logs
- **Latency**: Real network latency (100ms+)
- **Structured Output**: Valid JSON matching your models

### ⚠️ Warning Signs:
- Generic/repetitive responses
- Zero latency 
- Identical confidence scores
- No token usage reported

## 🎯 Production Testing Checklist

- [ ] **Basic LLM calls work**
- [ ] **Structured output parsing works** 
- [ ] **Error handling works** (try invalid prompts)
- [ ] **Rate limiting respected** (no 429 errors)
- [ ] **All agents can process real data**
- [ ] **Batch operations work**
- [ ] **Provider auto-selection works**
- [ ] **Helpful error messages appear**

## 💡 Testing Tips

1. **Start Small**: Test single LLM calls before complex agents
2. **Check Logs**: Look for real token usage and latency
3. **Rate Limits**: Gemini free tier has limits, space out tests
4. **Cost Awareness**: Real API calls cost money (though Gemini is generous)
5. **Error Testing**: Try invalid inputs to test error handling
6. **Provider Testing**: Test both Gemini and Claude if you have both keys

## 🔍 Debugging Commands

```bash
# Check what provider is being used
python -c "from src.utils.llm import create_llm_client; c = create_llm_client(); print('Provider:', type(c.provider).__name__)"

# Test structured output specifically  
python -c "
import asyncio
from src.utils.llm import create_llm_client
from src.critics.models import CriticScore

async def test():
    client = create_llm_client()
    result = await client.call('Rate this answer 1-100: Paris is the capital of France', response_format=CriticScore)
    print('Type:', type(result))
    print('Score:', result.overall_score if hasattr(result, 'overall_score') else 'No score')

asyncio.run(test())
"

# Test error handling
python -c "from src.utils.llm import create_llm_client; create_llm_client('invalid_provider')" 2>&1 || echo "✅ Error handling works"
```

---

## 🎉 Success Indicators

When everything works correctly, you should see:
- Real, intelligent responses to questions
- Proper JSON structure in structured outputs  
- Realistic token usage (not fixed numbers like 50/25)
- Network latencies (100ms+ typically)
- Provider type showing `GeminiLLMProvider`
- Varied confidence scores and content

Your system is now production-ready and will give you real AI-powered evaluations! 🚀