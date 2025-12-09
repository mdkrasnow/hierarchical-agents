"""
Tests for multi-critic debate and orchestration system.

Comprehensive test suite covering specialized critics, orchestration,
and debate aggregation functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from critics.multi_critic import (
    SpecializedCriticAgent, CoverageCritic, DepthCritic, StyleCritic, 
    InstructionCritic, CriticFactory, evaluate_with_all_critics
)
from critics.orchestrator import MultiCriticOrchestrator, ScoreAggregatorAgent
from critics.debate_models import (
    CriticRole, MultiCriticRequest, MultiCriticResult,
    CriticResult, ScoreAggregation, CriticConfiguration, AggregatorInput
)
from critics.models import CriticRequest
from critics.models import CriticScore, DimensionScore
from agents.base import AgentResult
from utils.llm import create_llm_client


class TestCriticRoleModel:
    """Test CriticRole model and validation."""
    
    def test_critic_role_creation(self):
        """Test creating a valid CriticRole."""
        role = CriticRole(
            role_name="coverage",
            display_name="Coverage Critic",
            description="Evaluates completeness",
            focus_areas=["completeness", "breadth"],
            template_name="coverage_critic",
            weight=0.3
        )
        
        assert role.role_name == "coverage"
        assert role.display_name == "Coverage Critic"
        assert role.weight == 0.3
        assert len(role.focus_areas) == 2
    
    def test_critic_role_weight_validation(self):
        """Test weight validation for CriticRole."""
        # Valid weights
        role1 = CriticRole(
            role_name="test", display_name="Test", description="Test",
            focus_areas=[], template_name="test", weight=0.0
        )
        assert role1.weight == 0.0
        
        role2 = CriticRole(
            role_name="test", display_name="Test", description="Test",
            focus_areas=[], template_name="test", weight=1.0
        )
        assert role2.weight == 1.0
        
        # Invalid weight should raise validation error
        with pytest.raises(ValueError):
            CriticRole(
                role_name="test", display_name="Test", description="Test",
                focus_areas=[], template_name="test", weight=1.5
            )


class TestSpecializedCritics:
    """Test specialized critic implementations."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for testing."""
        return create_llm_client("mock")
    
    @pytest.fixture
    def sample_critic_score(self):
        """Sample critic score for testing."""
        return CriticScore(
            overall_score=85,
            overall_tier="good",
            dimension_scores={
                "coverage": DimensionScore(
                    dimension_name="Information Coverage",
                    score=85,
                    tier="good",
                    justification="Good coverage of main topics",
                    weight=30,
                    weighted_score=25.5
                )
            },
            overall_justification="Well-covered answer with good detail",
            key_strengths=["Comprehensive coverage", "Clear structure"],
            key_weaknesses=["Could use more examples"],
            thinking_process=["Analyzed coverage", "Checked details"],
            rubric_version="1.0",
            evaluation_focus="coverage_completeness"
        )
    
    def test_coverage_critic_initialization(self, mock_llm_client):
        """Test CoverageCritic initialization."""
        critic = CoverageCritic(llm_client=mock_llm_client)
        
        assert critic.critic_role.role_name == "coverage"
        assert critic.critic_role.display_name == "Coverage Critic"
        assert critic.critic_role.weight == 0.3
        assert "completeness" in critic.critic_role.focus_areas
        assert critic.agent_type == "SpecializedCritic_coverage"
    
    def test_depth_critic_initialization(self, mock_llm_client):
        """Test DepthCritic initialization."""
        critic = DepthCritic(llm_client=mock_llm_client)
        
        assert critic.critic_role.role_name == "depth"
        assert critic.critic_role.display_name == "Depth Critic"
        assert critic.critic_role.weight == 0.25
        assert "technical_depth" in critic.critic_role.focus_areas
    
    def test_style_critic_initialization(self, mock_llm_client):
        """Test StyleCritic initialization."""
        critic = StyleCritic(llm_client=mock_llm_client)
        
        assert critic.critic_role.role_name == "style"
        assert critic.critic_role.display_name == "Style Critic"
        assert critic.critic_role.weight == 0.20
        assert "writing_clarity" in critic.critic_role.focus_areas
    
    def test_instruction_critic_initialization(self, mock_llm_client):
        """Test InstructionCritic initialization."""
        critic = InstructionCritic(llm_client=mock_llm_client)
        
        assert critic.critic_role.role_name == "instruction_following"
        assert critic.critic_role.display_name == "Instruction-Following Critic"
        assert critic.critic_role.weight == 0.15
        assert "requirement_compliance" in critic.critic_role.focus_areas
    
    @pytest.mark.asyncio
    async def test_specialized_critic_execution(self, mock_llm_client, sample_critic_score):
        """Test specialized critic execution."""
        # Mock successful LLM response
        mock_llm_client.call_async = AsyncMock(return_value=sample_critic_score)
        
        critic = CoverageCritic(llm_client=mock_llm_client)
        request = CriticRequest(
            question="What are the benefits of Python?",
            answer="Python is easy to learn and has good libraries.",
            context="Programming evaluation"
        )
        
        result = await critic.execute(request)
        
        assert result.success
        assert "critic_score" in result.data
        assert result.metadata["critic_role"] == "coverage"
        assert result.metadata["critic_display_name"] == "Coverage Critic"
        assert result.metadata["overall_score"] == 85
    
    @pytest.mark.asyncio
    async def test_specialized_critic_failure(self, mock_llm_client):
        """Test specialized critic handling of LLM failure."""
        # Mock LLM failure
        mock_llm_client.call_async = AsyncMock(side_effect=Exception("LLM failed"))
        
        critic = CoverageCritic(llm_client=mock_llm_client)
        request = CriticRequest(
            question="Test question",
            answer="Test answer"
        )
        
        result = await critic.execute(request)
        
        assert not result.success
        assert "LLM failed" in result.error
        assert result.metadata["critic_role"] == "coverage"


class TestCriticFactory:
    """Test CriticFactory functionality."""
    
    @pytest.fixture
    def mock_llm_client(self):
        return create_llm_client("mock")
    
    def test_get_available_roles(self):
        """Test getting available critic roles."""
        roles = CriticFactory.get_available_roles()
        
        expected_roles = ["coverage", "depth", "style", "instruction_following"]
        assert set(roles) == set(expected_roles)
    
    def test_create_coverage_critic(self, mock_llm_client):
        """Test creating coverage critic through factory."""
        critic = CriticFactory.create_critic("coverage", llm_client=mock_llm_client)
        
        assert isinstance(critic, CoverageCritic)
        assert critic.critic_role.role_name == "coverage"
    
    def test_create_invalid_critic(self, mock_llm_client):
        """Test creating invalid critic role."""
        with pytest.raises(ValueError, match="Unknown critic role 'invalid'"):
            CriticFactory.create_critic("invalid", llm_client=mock_llm_client)
    
    def test_create_all_critics(self, mock_llm_client):
        """Test creating all available critics."""
        critics = CriticFactory.create_all_critics(llm_client=mock_llm_client)
        
        expected_roles = ["coverage", "depth", "style", "instruction_following"]
        assert set(critics.keys()) == set(expected_roles)
        
        for role_name, critic in critics.items():
            assert isinstance(critic, SpecializedCriticAgent)
            assert critic.critic_role.role_name == role_name
    
    def test_get_default_configuration(self):
        """Test getting default system configuration."""
        config = CriticFactory.get_default_configuration()
        
        assert isinstance(config, CriticConfiguration)
        assert len(config.available_critics) == 4
        assert set(config.default_critics) == {"coverage", "depth", "style", "instruction_following"}
        assert config.enable_parallel_execution is True
        assert config.min_critics >= 1
        assert config.max_critics >= config.min_critics
        
        # Check that all critics have reasonable weights 
        for critic in config.available_critics:
            assert 0 < critic.weight <= 1.0


class TestScoreAggregatorAgent:
    """Test ScoreAggregatorAgent functionality."""
    
    @pytest.fixture
    def mock_llm_client(self):
        return create_llm_client("mock")
    
    @pytest.fixture
    def sample_aggregation(self):
        """Sample score aggregation for testing."""
        return ScoreAggregation(
            final_score=82,
            final_tier="good",
            aggregation_method="reasoned_synthesis",
            individual_scores={"coverage": 85, "depth": 80, "style": 81},
            score_variance=6.7,
            consensus_level="high",
            aggregation_reasoning="Critics showed strong agreement with minor variance",
            disagreement_analysis=["Minor disagreement on detail level"],
            consensus_points=["Good overall structure", "Appropriate coverage"],
            comprehensive_strengths=["Clear organization", "Good coverage"],
            comprehensive_weaknesses=["Could use more examples"],
            actionable_recommendations=["Add specific examples", "Enhance technical depth"]
        )
    
    @pytest.fixture
    def sample_critic_results(self):
        """Sample critic results for aggregation."""
        return [
            CriticResult(
                critic_role="coverage",
                critic_score=CriticScore(
                    overall_score=85,
                    overall_tier="good",
                    dimension_scores={},
                    overall_justification="Good coverage",
                    key_strengths=["Comprehensive"],
                    key_weaknesses=["Minor gaps"],
                    thinking_process=["Analyzed coverage"],
                    rubric_version="1.0",
                    evaluation_focus="coverage"
                ),
                execution_time_ms=1500,
                confidence=0.8,
                focus_summary="Coverage analysis",
                notable_observations=[]
            ),
            CriticResult(
                critic_role="depth",
                critic_score=CriticScore(
                    overall_score=80,
                    overall_tier="good",
                    dimension_scores={},
                    overall_justification="Good depth",
                    key_strengths=["Detailed"],
                    key_weaknesses=["Could be deeper"],
                    thinking_process=["Analyzed depth"],
                    rubric_version="1.0",
                    evaluation_focus="depth"
                ),
                execution_time_ms=1300,
                confidence=0.85,
                focus_summary="Depth analysis",
                notable_observations=[]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_aggregator_execution(self, mock_llm_client, sample_aggregation, sample_critic_results):
        """Test successful aggregator execution."""
        # Mock successful LLM response
        mock_llm_client.call_async = AsyncMock(return_value=sample_aggregation)
        
        aggregator = ScoreAggregatorAgent(llm_client=mock_llm_client)
        
        aggregator_input = AggregatorInput(
            question="Test question",
            answer="Test answer",
            context="Test context",
            critic_results=sample_critic_results,
            score_statistics={
                "mean": 82.5,
                "variance": 6.25,
                "count": 2
            }
        )
        
        result = await aggregator.execute(aggregator_input)
        
        assert result.success
        assert "score_aggregation" in result.data
        assert result.metadata["final_score"] == 82
        assert result.metadata["consensus_level"] == "high"
    
    @pytest.mark.asyncio
    async def test_aggregator_fallback(self, mock_llm_client, sample_critic_results):
        """Test aggregator fallback when LLM fails."""
        # Mock LLM failure
        mock_llm_client.call_async = AsyncMock(side_effect=Exception("LLM failed"))
        
        aggregator = ScoreAggregatorAgent(llm_client=mock_llm_client)
        
        aggregator_input = AggregatorInput(
            question="Test question",
            answer="Test answer",
            critic_results=sample_critic_results,
            score_statistics={
                "mean": 82.5,
                "variance": 6.25,
                "count": 2
            }
        )
        
        result = await aggregator.execute(aggregator_input)
        
        assert result.success
        aggregation = result.data["score_aggregation"]
        assert aggregation["aggregation_method"] == "fallback_averaging"
        assert aggregation["final_score"] == 82  # Average of 85 and 80


class TestMultiCriticOrchestrator:
    """Test MultiCriticOrchestrator functionality."""
    
    @pytest.fixture
    def mock_llm_client(self):
        return create_llm_client("mock")
    
    @pytest.fixture
    def sample_request(self):
        """Sample multi-critic request."""
        return MultiCriticRequest(
            question="What are the key features of Python?",
            answer="Python is a high-level programming language with dynamic typing and extensive libraries.",
            context="Programming evaluation",
            enable_debate=True,
            show_individual_results=True
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_basic_evaluation(self, mock_llm_client, sample_request):
        """Test basic orchestrator evaluation."""
        # Mock successful critic responses
        mock_critic_score = CriticScore(
            overall_score=85,
            overall_tier="good",
            dimension_scores={},
            overall_justification="Good answer",
            key_strengths=["Clear"],
            key_weaknesses=["Could improve"],
            thinking_process=["Analyzed"],
            rubric_version="1.0",
            evaluation_focus="test"
        )
        
        mock_aggregation = ScoreAggregation(
            final_score=85,
            final_tier="good",
            aggregation_method="reasoned_synthesis",
            individual_scores={"coverage": 85},
            score_variance=0,
            consensus_level="high",
            aggregation_reasoning="Single critic",
            disagreement_analysis=[],
            consensus_points=[],
            comprehensive_strengths=["Clear"],
            comprehensive_weaknesses=["Could improve"],
            actionable_recommendations=["Add examples"]
        )
        
        # Mock LLM calls
        mock_llm_client.call_async = AsyncMock()
        mock_llm_client.call_async.side_effect = [mock_critic_score, mock_aggregation]
        
        orchestrator = MultiCriticOrchestrator(mock_llm_client)
        
        # Limit to single critic for simpler test
        sample_request.critic_roles = ["coverage"]
        
        result = await orchestrator.evaluate(sample_request)
        
        assert isinstance(result, MultiCriticResult)
        assert result.final_score == 85
        assert result.final_tier == "good"
        assert len(result.debate_rounds) >= 1
        assert len(result.critics_used) == 1
        assert "coverage" in result.critics_used
    
    @pytest.mark.asyncio
    async def test_orchestrator_no_debate(self, mock_llm_client, sample_request):
        """Test orchestrator with debate disabled."""
        mock_critic_score = CriticScore(
            overall_score=80,
            overall_tier="good",
            dimension_scores={},
            overall_justification="Good answer",
            key_strengths=["Clear"],
            key_weaknesses=["Could improve"],
            thinking_process=["Analyzed"],
            rubric_version="1.0",
            evaluation_focus="test"
        )
        
        mock_llm_client.call_async = AsyncMock(return_value=mock_critic_score)
        
        orchestrator = MultiCriticOrchestrator(mock_llm_client)
        
        # Disable debate and use single critic
        sample_request.enable_debate = False
        sample_request.critic_roles = ["coverage"]
        
        result = await orchestrator.evaluate(sample_request)
        
        assert isinstance(result, MultiCriticResult)
        assert len(result.debate_rounds) == 1  # Only independent round
        assert result.debate_rounds[0].round_type == "independent"
        assert result.final_aggregation.aggregation_method == "simple_averaging"
    
    def test_orchestrator_invalid_critics(self, mock_llm_client, sample_request):
        """Test orchestrator with invalid critic roles."""
        sample_request.critic_roles = ["invalid_critic"]
        
        orchestrator = MultiCriticOrchestrator(mock_llm_client)
        
        with pytest.raises(RuntimeError, match="Multi-critic evaluation failed"):
            asyncio.run(orchestrator.evaluate(sample_request))


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def mock_llm_client(self):
        return create_llm_client("mock")
    
    @pytest.mark.asyncio
    async def test_full_multi_critic_workflow(self, mock_llm_client):
        """Test complete multi-critic evaluation workflow."""
        # Mock responses for all critics and aggregator
        mock_critic_score = CriticScore(
            overall_score=80,
            overall_tier="good",
            dimension_scores={
                "test_dim": DimensionScore(
                    dimension_name="Test Dimension",
                    score=80,
                    tier="good",
                    justification="Good performance",
                    weight=100,
                    weighted_score=80
                )
            },
            overall_justification="Good overall performance",
            key_strengths=["Clear structure"],
            key_weaknesses=["Could use more detail"],
            thinking_process=["Analyzed structure", "Checked detail"],
            rubric_version="1.0",
            evaluation_focus="test_focus"
        )
        
        mock_aggregation = ScoreAggregation(
            final_score=81,
            final_tier="good",
            aggregation_method="reasoned_synthesis",
            individual_scores={"coverage": 82, "depth": 80},
            score_variance=1.0,
            consensus_level="high",
            aggregation_reasoning="Strong consensus among critics",
            disagreement_analysis=["Minor variance in detail assessment"],
            consensus_points=["Good structure", "Clear presentation"],
            comprehensive_strengths=["Well organized", "Clear content"],
            comprehensive_weaknesses=["Could expand on examples"],
            actionable_recommendations=["Add specific examples", "Provide more detail"]
        )
        
        # Set up mock responses
        responses = [mock_critic_score] * 4 + [mock_aggregation]  # 4 critics + aggregator
        mock_llm_client.call_async = AsyncMock()
        mock_llm_client.call_async.side_effect = responses
        
        # Run evaluation
        request = MultiCriticRequest(
            question="Explain the benefits of object-oriented programming.",
            answer=("Object-oriented programming provides encapsulation, inheritance, "
                   "and polymorphism. These features help create modular, reusable code."),
            context="Software engineering education",
            enable_debate=True
        )
        
        orchestrator = MultiCriticOrchestrator(mock_llm_client)
        result = await orchestrator.evaluate(request)
        
        # Verify complete result structure
        assert isinstance(result, MultiCriticResult)
        assert result.final_score == 81
        assert result.final_tier == "good"
        assert len(result.debate_rounds) == 2  # Independent + aggregation
        assert len(result.critics_used) == 4
        assert result.confidence_level > 0
        assert result.total_execution_time_ms > 0
        
        # Verify rounds
        round1 = result.get_round_results(1)
        assert round1.round_type == "independent"
        assert len(round1.results) == 4
        
        round2 = result.get_round_results(2)
        assert round2.round_type == "aggregation"
        assert len(round2.results) == 1
        
        # Verify final aggregation
        assert result.final_aggregation.consensus_level == "high"
        assert len(result.final_aggregation.comprehensive_strengths) > 0
        assert len(result.final_aggregation.actionable_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_with_all_critics_convenience(self, mock_llm_client):
        """Test convenience function for evaluating with all critics."""
        mock_critic_score = CriticScore(
            overall_score=85,
            overall_tier="good",
            dimension_scores={},
            overall_justification="Good answer",
            key_strengths=["Clear"],
            key_weaknesses=["Minor issues"],
            thinking_process=["Analyzed"],
            rubric_version="1.0",
            evaluation_focus="test"
        )
        
        mock_llm_client.call_async = AsyncMock(return_value=mock_critic_score)
        
        results = await evaluate_with_all_critics(
            question="Test question",
            answer="Test answer",
            context="Test context",
            llm_client=mock_llm_client
        )
        
        assert isinstance(results, dict)
        assert len(results) == 4  # All critic roles
        
        expected_roles = {"coverage", "depth", "style", "instruction_following"}
        assert set(results.keys()) == expected_roles
        
        for role_name, score in results.items():
            assert isinstance(score, CriticScore)
            assert score.overall_score == 85


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def mock_llm_client(self):
        return create_llm_client("mock")
    
    @pytest.mark.asyncio
    async def test_partial_critic_failure(self, mock_llm_client):
        """Test handling when some critics fail but others succeed."""
        # Mock mixed responses - some succeed, some fail
        mock_critic_score = CriticScore(
            overall_score=80,
            overall_tier="good",
            dimension_scores={},
            overall_justification="Good",
            key_strengths=["Clear"],
            key_weaknesses=["Minor issues"],
            thinking_process=["Analyzed"],
            rubric_version="1.0",
            evaluation_focus="test"
        )
        
        responses = [
            mock_critic_score,  # Coverage succeeds
            Exception("Depth critic failed"),  # Depth fails
            mock_critic_score,  # Style succeeds  
            Exception("Instruction critic failed")  # Instruction fails
        ]
        
        mock_llm_client.call_async = AsyncMock()
        mock_llm_client.call_async.side_effect = responses
        
        results = await evaluate_with_all_critics(
            question="Test question",
            answer="Test answer",
            llm_client=mock_llm_client
        )
        
        # Should only have results from successful critics
        assert len(results) == 2
        assert "coverage" in results
        assert "style" in results
        assert "depth" not in results
        assert "instruction_following" not in results
    
    def test_critic_configuration_creation(self):
        """Test critic configuration creation."""
        # Test configuration with subset of critics
        test_roles = [
            CriticRole(role_name="test1", display_name="Test 1", description="Test",
                      focus_areas=[], template_name="test", weight=0.6),
            CriticRole(role_name="test2", display_name="Test 2", description="Test", 
                      focus_areas=[], template_name="test", weight=0.4)
        ]
        
        config = CriticConfiguration(
            available_critics=test_roles,
            default_critics=["test1", "test2"]
        )
        assert len(config.available_critics) == 2
        assert config.get_critic_role("test1") is not None
        assert config.get_critic_role("nonexistent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])