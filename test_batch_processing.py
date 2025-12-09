"""Test batch processing capabilities of the BaseAgent framework."""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.examples import BatchProcessingAgent
from utils.llm import create_llm_client


async def test_batch_processing():
    """Test batch processing functionality."""
    
    print("Testing BaseAgent batch processing...")
    
    # Create mock LLM client
    llm_client = create_llm_client(provider_type="mock", delay_ms=30)
    
    # Create batch processing agent
    agent = BatchProcessingAgent(llm_client=llm_client)
    
    print(f"Created agent: {agent}")
    
    # Test with multiple items
    test_items = [
        "Apple pie recipe",
        "Machine learning tutorial", 
        "Weather forecast for tomorrow",
        "Stock market analysis",
        "Movie review"
    ]
    
    print(f"\nProcessing {len(test_items)} items in batch...")
    
    result = await agent.execute_with_tracking(
        items=test_items,
        processing_type="categorize"
    )
    
    print(f"\nBatch processing result:")
    print(f"Success: {result.success}")
    print(f"Agent ID: {result.agent_id}")
    print(f"Execution time: {result.execution_time_ms}ms")
    
    if result.success and result.data:
        print(f"Processed: {result.data.get('processed_count', 0)} items")
        print(f"Failed: {result.data.get('failed_count', 0)} items")
        print(f"Total latency: {result.data.get('total_latency_ms', 0):.1f}ms")
        print(f"Avg per item: {result.data.get('total_latency_ms', 0) / len(test_items):.1f}ms")
        
        # Show first few results
        results = result.data.get('results', [])
        print(f"\nFirst few results:")
        for i, item_result in enumerate(results[:3]):
            print(f"  Item {i+1}: {item_result['item']}")
            print(f"    Result: {item_result['result'][:100]}...")
            print(f"    Latency: {item_result.get('latency_ms', 0):.1f}ms")
    
    if result.metrics:
        print(f"\nAgent metrics:")
        print(f"  LLM calls: {result.metrics.get('total_llm_calls', 0)}")
        print(f"  Total tokens: {result.metrics.get('total_llm_tokens', 0)}")
        print(f"  Avg latency: {result.metrics.get('avg_latency_ms', 0):.1f}ms")
        print(f"  Errors: {result.metrics.get('error_count', 0)}")
    
    print(f"\nAgent final status: {agent.status}")
    
    # Test empty batch
    print("\nTesting empty batch...")
    empty_result = await agent.execute_with_tracking(items=[])
    print(f"Empty batch success: {empty_result.success}")
    print(f"Empty batch processed: {empty_result.data.get('processed_count', 0) if empty_result.data else 0}")
    
    print("\nBatch processing test completed!")


if __name__ == "__main__":
    asyncio.run(test_batch_processing())