"""Basic test to validate the BaseAgent implementation."""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.examples import SimpleAnalyzerAgent
from agents.templates import FileTemplateLoader
from utils.llm import create_llm_client


async def test_base_agent_functionality():
    """Test basic agent functionality with mock LLM."""
    
    print("Testing BaseAgent framework...")
    
    # Create mock LLM client
    llm_client = create_llm_client(provider_type="mock", delay_ms=50)
    
    # Create template loader for our configs
    template_loader = FileTemplateLoader("configs/prompts")
    
    # Setup template manager
    from agents.templates import get_template_manager
    template_manager = get_template_manager()
    template_manager.add_loader("file", template_loader)
    
    # Create agent
    agent = SimpleAnalyzerAgent(
        llm_client=llm_client,
        template_manager=template_manager
    )
    
    print(f"Created agent: {agent}")
    print(f"Agent type: {agent.agent_type}")
    print(f"Role: {agent.role_description}")
    
    # Test execution
    test_text = "This is a sample text for analysis. It contains multiple sentences and should demonstrate the agent's capabilities."
    
    result = await agent.execute_with_tracking(
        text=test_text,
        analysis_type="content"
    )
    
    print(f"\nExecution result:")
    print(f"Success: {result.success}")
    print(f"Agent ID: {result.agent_id}")
    print(f"Execution time: {result.execution_time_ms}ms")
    
    if result.success:
        print(f"Analysis data keys: {list(result.data.keys()) if result.data else 'None'}")
        if result.metrics:
            print(f"LLM calls made: {result.metrics.get('total_llm_calls', 0)}")
            print(f"Average latency: {result.metrics.get('avg_latency_ms', 0):.1f}ms")
    else:
        print(f"Error: {result.error}")
    
    print(f"\nAgent final status: {agent.status}")
    print(f"Agent metrics: {agent._get_metrics_dict()}")
    
    print("\nTesting template rendering...")
    
    # Test template rendering
    try:
        prompt = await agent.render_prompt(
            "base_agent",
            {
                "responsibilities": "• Test responsibility 1\n• Test responsibility 2",
                "context": "Testing context",
                "additional_instructions": "This is a test."
            },
            loader_name="file"
        )
        print(f"Template rendered successfully. Length: {len(prompt)} characters")
        print(f"First 200 chars: {prompt[:200]}...")
    except Exception as e:
        print(f"Template rendering failed: {e}")
    
    print("\nBaseAgent framework test completed!")


if __name__ == "__main__":
    asyncio.run(test_base_agent_functionality())