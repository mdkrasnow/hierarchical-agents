"""
Comprehensive tests for the Danielson Evaluation Agent.

This module provides thorough testing of the evaluation agent including:
- Core functionality testing
- Edge case handling
- Missing data scenarios  
- Different framework configurations
- Risk detection accuracy
- Performance and scalability
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Dict, List
from uuid import uuid4

# Test imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.evaluation import DanielsonEvaluationAgent, EvaluationInput, RiskSignals
from agents.danielson import DanielsonSpecificAgent
from models import DanielsonEvaluation, EvaluationSummary, DomainSummary, DomainStatus
from utils.llm import create_llm_client


class TestDanielsonEvaluationAgent:
    """Test suite for DanielsonEvaluationAgent."""
    
    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client for testing."""
        return create_llm_client("mock", delay_ms=10, fail_rate=0.0)
    
    @pytest.fixture
    def agent(self, llm_client):
        """Create evaluation agent instance."""
        return DanielsonEvaluationAgent(llm_client=llm_client)
    
    @pytest.fixture
    def danielson_agent(self, llm_client):
        """Create Danielson-specific agent instance."""
        return DanielsonSpecificAgent(llm_client=llm_client)
    
    @pytest.fixture
    def org_config(self):
        """Standard organization configuration."""
        return {
            "version": "1.0",
            "framework": "danielson_2023",
            "domains": {
                "I-A": {"name": "Knowledge of Content and Pedagogy", "green": 3, "yellow": 2, "red": 1},
                "I-B": {"name": "Knowledge of Students", "green": 3, "yellow": 2, "red": 1},
                "I-C": {"name": "Setting Instructional Outcomes", "green": 3, "yellow": 2, "red": 1},
                "II-A": {"name": "Environment of Respect and Rapport", "green": 3, "yellow": 2, "red": 1},
                "II-B": {"name": "Culture for Learning", "green": 3, "yellow": 2, "red": 1},
                "III-A": {"name": "Communicating with Students", "green": 3, "yellow": 2, "red": 1},
                "III-B": {"name": "Using Questioning and Discussion", "green": 3, "yellow": 2, "red": 1},
                "III-C": {"name": "Engaging Students in Learning", "green": 3, "yellow": 2, "red": 1},
                "IV-A": {"name": "Reflecting on Teaching", "green": 3, "yellow": 2, "red": 1}
            },
            "global_thresholds": {
                "exemplar_teacher": 3.5,
                "proficient_teacher": 2.8,
                "developing_teacher": 2.0
            }
        }
    
    @pytest.fixture
    def sample_high_quality_evaluation(self):
        """High-quality evaluation with comprehensive data."""
        return DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Sarah Johnson",
            school_name="Lincoln Elementary",
            evaluator="Principal Smith",
            framework_id="danielson_2023",
            is_informal=False,
            created_at=datetime.now(),
            low_inference_notes="""
            Teacher demonstrated exceptional lesson planning and delivery. Students were highly engaged 
            throughout the 45-minute lesson on fractions. Clear learning objectives were posted and 
            referenced. Teacher used effective questioning strategies including wait time and follow-up 
            questions. Students responded with confidence and enthusiasm. Classroom environment was 
            positive with strong teacher-student rapport evident. Assessment was seamlessly integrated 
            with exit tickets checking for understanding.
            """,
            evaluation={
                "domains": {
                    "I-A": {"score": 4, "notes": "Exceptional content knowledge and pedagogical skills demonstrated"},
                    "I-C": {"score": 3, "notes": "Clear, measurable learning objectives aligned to standards"},
                    "II-A": {"score": 4, "notes": "Outstanding rapport with students, warm and respectful interactions"},
                    "II-B": {"score": 3, "notes": "Strong culture for learning, students motivated and engaged"},
                    "III-A": {"score": 3, "notes": "Clear communication, instructions well understood by all"},
                    "III-B": {"score": 4, "notes": "Sophisticated questioning with excellent wait time and probing"},
                    "III-C": {"score": 4, "notes": "Students deeply engaged in learning, active participation"},
                    "IV-A": {"score": 3, "notes": "Thoughtful reflection on lesson effectiveness and next steps"}
                }
            }
        )
    
    @pytest.fixture
    def sample_struggling_evaluation(self):
        """Evaluation showing teacher struggling with burnout signals."""
        return DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Mike Rodriguez",
            school_name="Oak Middle School", 
            evaluator="Assistant Principal Davis",
            framework_id="danielson_2023",
            is_informal=True,
            created_at=datetime.now(),
            low_inference_notes="""
            Teacher appeared overwhelmed and exhausted during this informal observation. Multiple 
            students were off-task and disruptive. Teacher raised voice several times to regain 
            attention. Lesson lacked clear structure and objectives. Teacher mentioned feeling 
            burned out and considering other career options during brief conversation. Students 
            seemed disengaged and confused by instructions. Classroom environment felt tense.
            """,
            evaluation={
                "domains": {
                    "I-A": {"score": 2, "notes": "Content knowledge adequate but delivery unclear"},
                    "II-A": {"score": 1, "notes": "Strained interactions, teacher frustration evident"},
                    "II-B": {"score": 1, "notes": "Poor learning environment, students disengaged"},
                    "III-A": {"score": 2, "notes": "Communication unclear, students frequently confused"},
                    "III-B": {"score": 1, "notes": "Minimal questioning, mostly teacher-directed lecture"},
                    "III-C": {"score": 1, "notes": "Low student engagement, many off-task behaviors"},
                    "IV-A": {"score": 2, "notes": "Teacher aware of issues but seems overwhelmed to address"}
                }
            }
        )
    
    @pytest.fixture 
    def sample_minimal_evaluation(self):
        """Evaluation with minimal data to test edge cases."""
        return DanielsonEvaluation(
            id=uuid4(),
            teacher_name=None,
            school_name=None,
            evaluation=None,
            low_inference_notes=None,
            created_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_basic_evaluation_processing(self, agent, sample_high_quality_evaluation, org_config):
        """Test basic evaluation processing with high-quality data."""
        
        eval_input = EvaluationInput(
            evaluation_data=sample_high_quality_evaluation,
            organization_config=org_config,
            include_evidence=True,
            max_evidence_snippets=10
        )
        
        result = await agent.execute(eval_input)
        
        assert result.success
        assert "evaluation_summary" in result.data
        
        # Validate EvaluationSummary structure
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        assert eval_summary.teacher_name == "Sarah Johnson"
        assert eval_summary.school_name == "Lincoln Elementary"
        assert eval_summary.evaluation_type == "formal"
        assert len(eval_summary.per_domain) > 0
        assert len(eval_summary.evidence_snippets) > 0
    
    @pytest.mark.asyncio
    async def test_domain_score_classification(self, agent, org_config):
        """Test domain score classification logic."""
        
        # Test different score values
        test_cases = [
            (4.0, DomainStatus.GREEN),
            (3.5, DomainStatus.GREEN), 
            (3.0, DomainStatus.GREEN),
            (2.5, DomainStatus.YELLOW),
            (2.0, DomainStatus.YELLOW),
            (1.5, DomainStatus.RED),
            (1.0, DomainStatus.RED),
            (None, DomainStatus.YELLOW)  # Missing data default
        ]
        
        for score, expected_status in test_cases:
            domain_config = {"green": 3, "yellow": 2, "red": 1}
            actual_status = agent._classify_domain_score(score, domain_config)
            assert actual_status == expected_status, f"Score {score} should be {expected_status}, got {actual_status}"
    
    @pytest.mark.asyncio
    async def test_risk_signal_detection(self, agent, sample_struggling_evaluation, org_config):
        """Test detection of burnout and disengagement signals."""
        
        eval_input = EvaluationInput(
            evaluation_data=sample_struggling_evaluation,
            organization_config=org_config
        )
        
        result = await agent.execute(eval_input)
        
        assert result.success
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # Should detect risk flags
        assert eval_summary.flags["burnout_signals"] == True
        assert eval_summary.flags["risk_of_leaving"] == True
        
        # Should have red/yellow domains
        red_domains = [d for d in eval_summary.per_domain.values() if d.status_color == DomainStatus.RED]
        assert len(red_domains) > 0
        
        # Should have concerns identified
        assert len(eval_summary.key_concerns) > 0
    
    @pytest.mark.asyncio
    async def test_missing_data_handling(self, agent, sample_minimal_evaluation, org_config):
        """Test graceful handling of missing or minimal data."""
        
        eval_input = EvaluationInput(
            evaluation_data=sample_minimal_evaluation,
            organization_config=org_config
        )
        
        result = await agent.execute(eval_input)
        
        assert result.success
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # Should handle missing names gracefully
        assert eval_summary.teacher_name == "Unknown Teacher"
        assert eval_summary.school_name == "Unknown School"
        
        # Should still have domain structure with defaults
        assert len(eval_summary.per_domain) > 0
        
        # Should have low relevance
        assert eval_summary.relevance_to_question == "low"
    
    @pytest.mark.asyncio
    async def test_evidence_extraction(self, agent, sample_high_quality_evaluation, org_config):
        """Test evidence snippet extraction functionality."""
        
        eval_input = EvaluationInput(
            evaluation_data=sample_high_quality_evaluation,
            organization_config=org_config,
            include_evidence=True,
            max_evidence_snippets=5
        )
        
        result = await agent.execute(eval_input)
        
        assert result.success
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # Should extract evidence
        assert len(eval_summary.evidence_snippets) > 0
        assert len(eval_summary.evidence_snippets) <= 5  # Respects max limit
        
        # Evidence should be meaningful (not too short)
        for evidence in eval_summary.evidence_snippets:
            assert len(evidence.strip()) > 15
    
    @pytest.mark.asyncio
    async def test_flag_generation(self, agent, sample_high_quality_evaluation, org_config):
        """Test flag generation logic."""
        
        eval_input = EvaluationInput(
            evaluation_data=sample_high_quality_evaluation,
            organization_config=org_config
        )
        
        result = await agent.execute(eval_input)
        
        assert result.success
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # High-quality evaluation should show exemplar potential
        flags = eval_summary.flags
        assert "exemplar" in flags
        assert "needs_PD" in flags
        assert "burnout_signals" in flags
        assert "risk_of_leaving" in flags
        
        # Should not have burnout/leaving risk for high-quality eval
        assert flags["burnout_signals"] == False
        assert flags["risk_of_leaving"] == False
    
    @pytest.mark.asyncio
    async def test_custom_org_thresholds(self, agent, sample_high_quality_evaluation):
        """Test custom organization threshold configurations."""
        
        # Custom config with higher thresholds
        custom_config = {
            "domains": {
                "I-A": {"green": 3.5, "yellow": 2.5, "red": 1.5},
                "III-B": {"green": 3.5, "yellow": 2.5, "red": 1.5}
            },
            "global_thresholds": {
                "exemplar_teacher": 4.0  # Very high threshold
            }
        }
        
        eval_input = EvaluationInput(
            evaluation_data=sample_high_quality_evaluation,
            organization_config=custom_config
        )
        
        result = await agent.execute(eval_input)
        
        assert result.success
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # With higher thresholds, some domains might be yellow instead of green
        domain_statuses = [d.status_color for d in eval_summary.per_domain.values() if d.score is not None]
        assert DomainStatus.YELLOW in domain_statuses or DomainStatus.GREEN in domain_statuses
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, agent, org_config):
        """Test performance with multiple evaluations processed concurrently."""
        
        # Create multiple evaluation inputs
        eval_inputs = []
        for i in range(10):
            evaluation = DanielsonEvaluation(
                id=uuid4(),
                teacher_name=f"Teacher {i}",
                school_name="Test School",
                created_at=datetime.now(),
                evaluation={"domains": {"I-A": {"score": 2 + (i % 3), "notes": f"Notes for teacher {i}"}}}
            )
            eval_inputs.append(EvaluationInput(
                evaluation_data=evaluation,
                organization_config=org_config
            ))
        
        # Process concurrently
        tasks = [agent.execute(eval_input) for eval_input in eval_inputs]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result.success for result in results)
        assert len(results) == 10


class TestDanielsonSpecificAgent:
    """Test suite for Danielson-specific agent enhancements."""
    
    @pytest.fixture
    def llm_client(self):
        return create_llm_client("mock", delay_ms=10)
    
    @pytest.fixture
    def danielson_agent(self, llm_client):
        return DanielsonSpecificAgent(llm_client=llm_client)
    
    @pytest.fixture
    def org_config(self):
        return {
            "domains": {
                "I-A": {"green": 3, "yellow": 2, "red": 1},
                "III-B": {"green": 3, "yellow": 2, "red": 1},
                "III-C": {"green": 3, "yellow": 2, "red": 1}
            }
        }
    
    @pytest.mark.asyncio
    async def test_priority_score_calculation(self, danielson_agent):
        """Test Danielson priority score calculation."""
        
        # Create domain summaries with different scores
        domain_summaries = {
            "III-B": DomainSummary(domain_id="III-B", score=4.0, status_color=DomainStatus.GREEN),
            "III-C": DomainSummary(domain_id="III-C", score=3.0, status_color=DomainStatus.GREEN),
            "I-A": DomainSummary(domain_id="I-A", score=2.0, status_color=DomainStatus.YELLOW),
            "Unknown": DomainSummary(domain_id="Unknown", score=1.0, status_color=DomainStatus.RED)
        }
        
        priority_score = await danielson_agent.calculate_danielson_priority_score(domain_summaries)
        
        # Should be weighted average of known domains
        assert 2.0 < priority_score < 4.0  # Between min and max known scores
    
    @pytest.mark.asyncio
    async def test_focus_domains_identification(self, danielson_agent):
        """Test identification of focus domains for PD."""
        
        domain_summaries = {
            "III-B": DomainSummary(
                domain_id="III-B", 
                score=1.0, 
                status_color=DomainStatus.RED,
                summary="Needs significant improvement in questioning"
            ),
            "III-C": DomainSummary(
                domain_id="III-C", 
                score=2.0, 
                status_color=DomainStatus.YELLOW,
                summary="Some engagement strategies present"
            ),
            "I-A": DomainSummary(
                domain_id="I-A", 
                score=4.0, 
                status_color=DomainStatus.GREEN,
                summary="Strong content knowledge"
            )
        }
        
        focus_domains = await danielson_agent.get_danielson_focus_domains(domain_summaries, max_domains=2)
        
        # Should prioritize red domains first
        assert len(focus_domains) <= 2
        assert focus_domains[0][0] == "III-B"  # Red domain should be first
        assert "Critical need" in focus_domains[0][1]
    
    @pytest.mark.asyncio
    async def test_instruction_domain_analysis(self, danielson_agent, org_config):
        """Test enhanced analysis of core instruction domains."""
        
        evaluation = DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Test Teacher",
            created_at=datetime.now(),
            evaluation={
                "domains": {
                    "III-B": {
                        "score": 4, 
                        "notes": "Teacher used effective questioning strategies with excellent wait time"
                    },
                    "III-C": {
                        "score": 3,
                        "notes": "Students were engaged in hands-on learning activities"
                    }
                }
            }
        )
        
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config
        )
        
        result = await danielson_agent.execute(eval_input)
        
        assert result.success
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # Should have enhanced analysis for instruction domains
        iii_b_domain = eval_summary.per_domain.get("III-B")
        if iii_b_domain:
            # Should detect questioning evidence
            growth_signals_text = " ".join(iii_b_domain.growth_signals).lower()
            assert "question" in growth_signals_text or len(iii_b_domain.evidence_quotes) > 0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def llm_client(self):
        # Use mock client with some failure rate for testing
        return create_llm_client("mock", fail_rate=0.1, delay_ms=5)
    
    @pytest.fixture
    def agent(self, llm_client):
        return DanielsonEvaluationAgent(llm_client=llm_client)
    
    @pytest.mark.asyncio
    async def test_malformed_evaluation_json(self, agent):
        """Test handling of malformed evaluation JSON data."""
        
        evaluation = DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Test Teacher",
            created_at=datetime.now(),
            evaluation={"malformed": "data", "nested": {"broken": None}}
        )
        
        org_config = {"domains": {"I-A": {"green": 3, "yellow": 2, "red": 1}}}
        
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config
        )
        
        # Should not crash, should handle gracefully
        result = await agent.execute(eval_input)
        assert result.success  # Should succeed even with malformed data
    
    @pytest.mark.asyncio
    async def test_empty_org_config(self, agent):
        """Test behavior with empty organization configuration."""
        
        evaluation = DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Test Teacher",
            created_at=datetime.now()
        )
        
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config={}  # Empty config
        )
        
        result = await agent.execute(eval_input)
        assert result.success  # Should use defaults
    
    @pytest.mark.asyncio
    async def test_extreme_text_lengths(self, agent):
        """Test handling of very long or very short text inputs."""
        
        # Very long notes
        long_notes = "This is a test note. " * 1000  # Very long
        
        evaluation = DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Test Teacher",
            created_at=datetime.now(),
            low_inference_notes=long_notes
        )
        
        org_config = {"domains": {"I-A": {"green": 3, "yellow": 2, "red": 1}}}
        
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config,
            max_evidence_snippets=20
        )
        
        result = await agent.execute(eval_input)
        assert result.success
        
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        # Should limit evidence appropriately
        assert len(eval_summary.evidence_snippets) <= 20
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, agent):
        """Test handling of unicode and special characters."""
        
        evaluation = DanielsonEvaluation(
            id=uuid4(),
            teacher_name="José María García-López",
            school_name="École Élémentaire",
            created_at=datetime.now(),
            low_inference_notes="Teacher showed émotions and used café-style discussion circles. Students responded with enthusiasm! 🎉"
        )
        
        org_config = {"domains": {"I-A": {"green": 3, "yellow": 2, "red": 1}}}
        
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config
        )
        
        result = await agent.execute(eval_input)
        assert result.success
        
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        assert "José" in eval_summary.teacher_name
        assert "École" in eval_summary.school_name


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def llm_client(self):
        return create_llm_client("mock", delay_ms=25)
    
    @pytest.fixture
    def agent(self, llm_client):
        return DanielsonEvaluationAgent(llm_client=llm_client)
    
    @pytest.mark.asyncio
    async def test_real_world_evaluation_scenario(self, agent):
        """Test with realistic evaluation data and configuration."""
        
        # Realistic evaluation data
        evaluation = DanielsonEvaluation(
            id=uuid4(),
            teacher_name="Amanda Williams",
            school_name="Roosevelt Elementary School",
            evaluator="Dr. Patricia Johnson, Principal",
            framework_id="danielson_2014_updated",
            is_informal=False,
            created_at=datetime.now(),
            low_inference_notes="""
            Formal observation of 4th grade mathematics lesson on multiplication strategies.
            Lesson objective: Students will use multiple strategies to solve two-digit multiplication problems.
            
            Teacher began with number talk routine, asking students to solve 12 x 15 mentally using different approaches.
            Students shared strategies including breaking apart numbers, using arrays, and traditional algorithm.
            Teacher asked probing questions: "Why did you choose that strategy?" and "How does your answer compare to Maya's?"
            
            During independent work time, teacher circulated and provided differentiated support.
            Noticed teacher gave struggling students manipulatives and visual supports.
            Advanced students received extension problems involving three-digit numbers.
            
            Classroom environment was positive - students felt comfortable sharing mistakes and learning from them.
            Teacher used positive language: "That's interesting thinking" and "Let's build on Marcus's idea."
            
            Assessment: Teacher used exit ticket to gauge understanding. Also observed informal assessment through questioning.
            Some students still struggling with regrouping concept - teacher noted this for follow-up instruction.
            
            Areas for growth: Could incorporate more movement/kinesthetic activities for engagement.
            Teacher reflection was thoughtful - identified next steps for differentiation.
            """,
            evaluation={
                "domains": {
                    "I-A": {"score": 3, "notes": "Strong mathematical content knowledge. Understands grade-level concepts and multiple solution strategies."},
                    "I-B": {"score": 3, "notes": "Knows students' mathematical thinking patterns. Provides appropriate differentiation."},
                    "I-C": {"score": 3, "notes": "Clear, measurable learning objective aligned to standards. Posted and referenced during lesson."},
                    "II-A": {"score": 4, "notes": "Excellent rapport with students. Creates safe environment for risk-taking and mistake-making."},
                    "II-B": {"score": 3, "notes": "Strong culture for learning. Students engaged and motivated to participate."},
                    "III-A": {"score": 3, "notes": "Clear communication. Instructions understood by students at different levels."},
                    "III-B": {"score": 4, "notes": "Excellent questioning strategies. Uses wait time, probing questions, builds on student responses."},
                    "III-C": {"score": 3, "notes": "Good engagement strategies. Could incorporate more movement and hands-on activities."},
                    "III-D": {"score": 3, "notes": "Uses formative assessment effectively. Exit tickets and questioning to gauge understanding."},
                    "IV-A": {"score": 3, "notes": "Thoughtful reflection on lesson effectiveness. Identified areas for improvement and next steps."}
                }
            }
        )
        
        # Realistic org config
        org_config = {
            "version": "2.0",
            "framework": "danielson_2014_updated",
            "domains": {
                "I-A": {"name": "Demonstrating Knowledge of Content and Pedagogy", "green": 3, "yellow": 2, "red": 1},
                "I-B": {"name": "Demonstrating Knowledge of Students", "green": 3, "yellow": 2, "red": 1},
                "I-C": {"name": "Setting Instructional Outcomes", "green": 3, "yellow": 2, "red": 1},
                "II-A": {"name": "Creating Environment of Respect and Rapport", "green": 3, "yellow": 2, "red": 1},
                "II-B": {"name": "Establishing Culture for Learning", "green": 3, "yellow": 2, "red": 1},
                "III-A": {"name": "Communicating with Students", "green": 3, "yellow": 2, "red": 1},
                "III-B": {"name": "Using Questioning and Discussion Techniques", "green": 3, "yellow": 2, "red": 1},
                "III-C": {"name": "Engaging Students in Learning", "green": 3, "yellow": 2, "red": 1},
                "III-D": {"name": "Using Assessment in Instruction", "green": 3, "yellow": 2, "red": 1},
                "IV-A": {"name": "Reflecting on Teaching", "green": 3, "yellow": 2, "red": 1}
            },
            "global_thresholds": {
                "exemplar_teacher": 3.6,
                "proficient_teacher": 2.8,
                "developing_teacher": 2.0,
                "ineffective_teacher": 1.5
            },
            "risk_indicators": {
                "high_risk": {"red_domain_count": 2, "declining_trend_months": 6},
                "medium_risk": {"yellow_domain_count": 3, "stagnant_trend_months": 12}
            }
        }
        
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config,
            analysis_focus="comprehensive",
            include_evidence=True,
            max_evidence_snippets=15
        )
        
        result = await agent.execute(eval_input)
        
        # Comprehensive validation
        assert result.success
        assert result.data is not None
        
        eval_summary = EvaluationSummary(**result.data["evaluation_summary"])
        
        # Teacher info
        assert eval_summary.teacher_name == "Amanda Williams"
        assert eval_summary.school_name == "Roosevelt Elementary School"
        assert eval_summary.evaluation_type == "formal"
        
        # Should detect high quality performance
        green_domains = [d for d in eval_summary.per_domain.values() 
                        if d.status_color == DomainStatus.GREEN and d.score is not None]
        assert len(green_domains) >= 5  # Most domains should be green
        
        # Should extract meaningful evidence
        assert len(eval_summary.evidence_snippets) > 0
        evidence_text = " ".join(eval_summary.evidence_snippets).lower()
        assert any(keyword in evidence_text for keyword in ["question", "student", "learning"])
        
        # Should identify strengths
        assert len(eval_summary.key_strengths) > 0
        strengths_text = " ".join(eval_summary.key_strengths).lower()
        assert any(keyword in strengths_text for keyword in ["question", "rapport", "environment"])
        
        # Should not detect major risk signals for this high-performing teacher
        assert eval_summary.flags["burnout_signals"] == False
        assert eval_summary.flags["risk_of_leaving"] == False
        
        # Should have high relevance
        assert eval_summary.relevance_to_question in ["high", "medium"]


# Helper functions for test data creation
def create_test_evaluation(
    teacher_name: str = "Test Teacher",
    domain_scores: Dict[str, float] = None,
    notes: str = "Test evaluation notes",
    is_informal: bool = False
) -> DanielsonEvaluation:
    """Create test evaluation with specified parameters."""
    
    if domain_scores is None:
        domain_scores = {"I-A": 3.0, "III-B": 2.0}
    
    evaluation_data = {
        "domains": {
            domain_id: {"score": score, "notes": f"Notes for {domain_id}"}
            for domain_id, score in domain_scores.items()
        }
    }
    
    return DanielsonEvaluation(
        id=uuid4(),
        teacher_name=teacher_name,
        school_name="Test School",
        is_informal=is_informal,
        created_at=datetime.now(),
        low_inference_notes=notes,
        evaluation=evaluation_data
    )


# Performance and stress testing
@pytest.mark.performance  
class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batches of evaluations."""
        
        llm_client = create_llm_client("mock", delay_ms=1)  # Fast mock
        agent = DanielsonEvaluationAgent(llm_client=llm_client)
        
        # Create 50 test evaluations
        evaluations = [
            create_test_evaluation(f"Teacher {i}", {"I-A": 2 + (i % 3), "III-B": 1 + (i % 4)})
            for i in range(50)
        ]
        
        org_config = {"domains": {"I-A": {"green": 3, "yellow": 2, "red": 1}}}
        
        eval_inputs = [
            EvaluationInput(evaluation_data=eval_data, organization_config=org_config)
            for eval_data in evaluations
        ]
        
        # Time the processing
        import time
        start_time = time.time()
        
        tasks = [agent.execute(eval_input) for eval_input in eval_inputs]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Validate results
        successful = sum(1 for r in results if r.success)
        assert successful >= 45  # Allow some failures with mock client
        
        # Performance check - should process reasonably fast
        total_time = end_time - start_time
        assert total_time < 30  # Should complete within 30 seconds
        
        avg_time_per_eval = total_time / len(evaluations)
        assert avg_time_per_eval < 1  # Less than 1 second per evaluation


if __name__ == "__main__":
    # Run specific test if called directly
    pytest.main([__file__, "-v"])