"""
Test suite for answer generation system.

Tests the complete answer generation pipeline including LLM and hierarchical
generation strategies, validation, and integration with the scoring system.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import (
    AnswerGenerator, GenerationConfig, GenerationResult, GenerationStrategy,
    LLMGenerator, HierarchicalGenerator, create_generator, generate_answer_simple
)
from utils.llm import create_llm_client, LLMResponse


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.strategy == GenerationStrategy.LLM_GENERIC
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.use_chain_of_thought == False
        assert config.enable_validation == True
        assert config.max_retries == 3
        assert isinstance(config.generation_context, dict)
        assert isinstance(config.agent_types, list)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            strategy=GenerationStrategy.HIERARCHICAL_AGENT,
            temperature=0.2,
            max_tokens=1024,
            use_chain_of_thought=True,
            custom_instructions="Be very detailed",
            generation_context={"domain": "education"}
        )
        
        assert config.strategy == GenerationStrategy.HIERARCHICAL_AGENT
        assert config.temperature == 0.2
        assert config.max_tokens == 1024
        assert config.use_chain_of_thought == True
        assert config.custom_instructions == "Be very detailed"
        assert config.generation_context["domain"] == "education"


class TestGenerationResult:
    """Test GenerationResult model."""
    
    def test_successful_result(self):
        """Test successful generation result."""
        result = GenerationResult(
            answer="This is a test answer",
            success=True,
            strategy_used="test_strategy",
            generation_time_ms=150.0,
            confidence_score=0.85
        )
        
        assert result.success == True
        assert result.answer == "This is a test answer"
        assert result.strategy_used == "test_strategy"
        assert result.generation_time_ms == 150.0
        assert result.confidence_score == 0.85
        assert len(result.generation_id) > 0
    
    def test_failed_result(self):
        """Test failed generation result."""
        result = GenerationResult(
            answer="",
            success=False,
            strategy_used="test_strategy",
            generation_time_ms=50.0,
            error_message="Generation failed"
        )
        
        assert result.success == False
        assert result.error_message == "Generation failed"
    
    def test_result_methods(self):
        """Test result helper methods."""
        result = GenerationResult(
            answer="Test",
            success=True,
            strategy_used="test",
            generation_time_ms=100.0
        )
        
        result.add_warning("Test warning")
        result.add_reasoning_step("Step 1")
        result.set_error("Test error")
        
        assert "Test warning" in result.warnings
        assert "Step 1" in result.reasoning_trace
        assert result.success == False
        assert result.error_message == "Test error"


class TestLLMGenerator:
    """Test LLM-based answer generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_client = Mock()
        self.config = GenerationConfig(temperature=0.5)
        self.generator = LLMGenerator(llm_client=self.mock_llm_client, config=self.config)
    
    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Test basic answer generation."""
        # Mock LLM response
        mock_response = LLMResponse(
            content="This is a generated answer",
            latency_ms=100.0,
            token_usage={"input_tokens": 50, "output_tokens": 25}
        )
        self.mock_llm_client.call = AsyncMock(return_value=mock_response)
        
        question = "What is the capital of France?"
        result = await self.generator.generate_answer(question)
        
        assert result.success == True
        assert result.answer == "This is a generated answer"
        assert result.strategy_used == "llm_generic"
        assert result.llm_calls_made == 1
        assert result.total_tokens_used == 75
    
    @pytest.mark.asyncio
    async def test_chain_of_thought_generation(self):
        """Test chain-of-thought generation."""
        config = GenerationConfig(use_chain_of_thought=True)
        generator = LLMGenerator(llm_client=self.mock_llm_client, config=config)
        
        # Mock LLM response with CoT format
        mock_response = LLMResponse(
            content="Reasoning:\n1. France is a country in Europe\n2. Paris is the largest city\nFinal Answer: The capital of France is Paris",
            latency_ms=150.0
        )
        self.mock_llm_client.call = AsyncMock(return_value=mock_response)
        
        question = "What is the capital of France?"
        result = await generator.generate_answer(question)
        
        assert result.success == True
        assert "Paris" in result.answer
        assert result.strategy_used == "llm_chain_of_thought"
        assert len(result.reasoning_trace) > 0
    
    @pytest.mark.asyncio
    async def test_generation_with_context(self):
        """Test generation with additional context."""
        mock_response = LLMResponse(content="Answer with context", latency_ms=100.0)
        self.mock_llm_client.call = AsyncMock(return_value=mock_response)
        
        question = "Analyze this situation"
        context = {"domain": "education", "level": "high_school"}
        
        result = await self.generator.generate_answer(question, context=context)
        
        assert result.success == True
        # Verify context was passed through
        call_args = self.mock_llm_client.call.call_args[1]
        assert "education" in call_args["prompt"]
    
    @pytest.mark.asyncio
    async def test_generation_failure(self):
        """Test handling of generation failures."""
        self.mock_llm_client.call = AsyncMock(side_effect=Exception("API error"))
        
        question = "Test question"
        result = await self.generator.generate_answer(question)
        
        assert result.success == False
        assert "API error" in result.error_message
    
    @pytest.mark.asyncio
    async def test_multiple_answers_generation(self):
        """Test generating multiple answers."""
        mock_response = LLMResponse(content="Generated answer", latency_ms=100.0)
        self.mock_llm_client.call = AsyncMock(return_value=mock_response)
        
        questions = ["Question 1", "Question 2", "Question 3"]
        results = await self.generator.generate_multiple_answers(questions)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert self.mock_llm_client.call.call_count == 3
    
    @pytest.mark.asyncio
    async def test_answer_validation(self):
        """Test answer validation."""
        # Test valid answer
        validation = await self.generator.validate_answer(
            "What is 2+2?", "2+2 equals 4 because addition combines values."
        )
        assert validation["is_valid"] == True
        assert validation["quality_score"] > 0.5
        
        # Test empty answer
        validation = await self.generator.validate_answer("What is 2+2?", "")
        assert validation["is_valid"] == False
        assert "empty" in validation["issues"][0].lower()
        
        # Test very short answer
        validation = await self.generator.validate_answer("What is 2+2?", "4")
        assert len(validation["issues"]) > 0
        assert "short" in validation["issues"][0].lower()
    
    def test_strategy_properties(self):
        """Test generator strategy properties."""
        assert self.generator.strategy_name == "llm_generic"
        assert isinstance(self.generator.supported_question_types, list)
        assert "general-writing" in self.generator.supported_question_types


class TestHierarchicalGenerator:
    """Test hierarchical agent-based answer generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_client = Mock()
        self.config = GenerationConfig(strategy=GenerationStrategy.HIERARCHICAL_AGENT)
        self.generator = HierarchicalGenerator(llm_client=self.mock_llm_client, config=self.config)
    
    @pytest.mark.asyncio
    async def test_question_analysis(self):
        """Test question analysis for agent selection."""
        # Mock the analysis method to test different question types
        danielson_question = "Analyze the teacher's performance using Danielson framework"
        strategy = await self.generator._analyze_question_for_agents(
            danielson_question, {}, self.config, Mock()
        )
        
        assert strategy is not None
        assert strategy["type"] in ["single_agent", "multi_agent"]
    
    def test_strategy_properties(self):
        """Test hierarchical generator properties."""
        assert self.generator.strategy_name == "hierarchical_agent"
        assert isinstance(self.generator.supported_question_types, list)
        assert "evaluation-analysis" in self.generator.supported_question_types


class TestGeneratorFactory:
    """Test generator creation utilities."""
    
    def test_create_generator_llm(self):
        """Test creating LLM generator."""
        generator = create_generator(GenerationStrategy.LLM_GENERIC)
        assert isinstance(generator, LLMGenerator)
        assert generator.strategy_name == "llm_generic"
    
    def test_create_generator_hierarchical(self):
        """Test creating hierarchical generator.""" 
        generator = create_generator(GenerationStrategy.HIERARCHICAL_AGENT)
        assert isinstance(generator, HierarchicalGenerator)
        assert generator.strategy_name == "hierarchical_agent"
    
    def test_create_generator_with_config(self):
        """Test creating generator with custom config."""
        config = GenerationConfig(temperature=0.2, max_tokens=512)
        generator = create_generator("llm_generic", config=config)
        
        assert generator.config.temperature == 0.2
        assert generator.config.max_tokens == 512
    
    def test_unsupported_strategy(self):
        """Test error handling for unsupported strategy."""
        with pytest.raises(ValueError):
            create_generator("unsupported_strategy")
    
    @pytest.mark.asyncio
    async def test_simple_generation(self):
        """Test simple answer generation utility."""
        with patch('generation.answer_generator.create_generator') as mock_create:
            mock_generator = Mock()
            mock_result = GenerationResult(
                answer="Simple answer",
                success=True,
                strategy_used="test",
                generation_time_ms=100.0
            )
            mock_generator.generate_answer = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_generator
            
            answer = await generate_answer_simple("Test question")
            assert answer == "Simple answer"
    
    @pytest.mark.asyncio
    async def test_simple_generation_failure(self):
        """Test simple generation with failure."""
        with patch('generation.answer_generator.create_generator') as mock_create:
            mock_generator = Mock()
            mock_result = GenerationResult(
                answer="",
                success=False,
                strategy_used="test",
                generation_time_ms=0.0,
                error_message="Failed"
            )
            mock_generator.generate_answer = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_generator
            
            with pytest.raises(RuntimeError):
                await generate_answer_simple("Test question")


class TestAnswerGenerationIntegration:
    """Integration tests for answer generation system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_llm_generation(self):
        """Test complete LLM generation pipeline."""
        # Use mock LLM client
        llm_client = create_llm_client("mock", delay_ms=10)
        
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=512,
            enable_validation=True
        )
        
        generator = LLMGenerator(llm_client=llm_client, config=config)
        
        question = "Explain the importance of teacher-student relationships in education."
        result = await generator.generate_answer(question)
        
        assert result.success == True
        assert len(result.answer) > 0
        assert result.generation_time_ms > 0
        assert result.confidence_score is not None
    
    @pytest.mark.asyncio 
    async def test_end_to_end_hierarchical_generation(self):
        """Test complete hierarchical generation pipeline."""
        # Use mock LLM client
        llm_client = create_llm_client("mock", delay_ms=10)
        
        config = GenerationConfig(
            strategy=GenerationStrategy.HIERARCHICAL_AGENT,
            temperature=0.5
        )
        
        generator = HierarchicalGenerator(llm_client=llm_client, config=config)
        
        # Use an evaluation-style question
        question = "Analyze this teacher's classroom management strategies and provide recommendations."
        context = {
            "teacher_name": "Ms. Smith",
            "school_name": "Test Elementary"
        }
        
        result = await generator.generate_answer(question, context=context)
        
        # Should succeed with mock agents
        assert result.success == True
        assert len(result.answer) > 0
    
    @pytest.mark.asyncio
    async def test_config_override(self):
        """Test configuration override functionality."""
        llm_client = create_llm_client("mock", delay_ms=5)
        
        base_config = GenerationConfig(temperature=0.7, max_tokens=1024)
        generator = LLMGenerator(llm_client=llm_client, config=base_config)
        
        override_config = GenerationConfig(temperature=0.2, use_chain_of_thought=True)
        
        question = "Test question"
        result = await generator.generate_answer(question, config_override=override_config)
        
        assert result.success == True
        # Should use chain-of-thought from override
        assert result.strategy_used == "llm_chain_of_thought"


if __name__ == "__main__":
    # Run specific test if specified, otherwise run all tests
    if len(sys.argv) > 1:
        pytest.main(["-v", f"test_{sys.argv[1]}"])
    else:
        pytest.main(["-v", __file__])