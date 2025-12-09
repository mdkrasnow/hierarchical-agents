#!/usr/bin/env python3
"""
Script to run the Danielson Evaluation Agent for testing and demonstration.

This script provides examples of how to use the evaluation agent with
sample data, different configurations, and various testing scenarios.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.evaluation import DanielsonEvaluationAgent, EvaluationInput
from agents.danielson import DanielsonSpecificAgent
from models import DanielsonEvaluation, EvaluationSummary
from utils.llm import create_llm_client, LLMClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_evaluation_data() -> List[Dict]:
    """Create sample evaluation data for testing."""
    
    sample_evaluations = [
        {
            "id": str(uuid4()),
            "teacher_name": "Sarah Johnson",
            "school_name": "Lincoln Elementary",
            "evaluator": "Principal Smith", 
            "framework_id": "danielson_2023",
            "is_informal": False,
            "created_at": datetime.now(),
            "low_inference_notes": """
            Teacher demonstrated clear learning objectives posted on board. Students were actively engaged 
            in small group discussions about the water cycle. Saw effective use of questioning - teacher 
            asked follow-up questions to deepen understanding. Classroom environment was positive with 
            respectful interactions between teacher and students. Some students seemed to struggle with 
            the vocabulary, but teacher provided additional support. Overall strong lesson with good 
            student engagement.
            """,
            "evaluation": {
                "domains": {
                    "I-A": {"score": 3, "notes": "Strong content knowledge demonstrated. Clear understanding of science concepts."},
                    "I-C": {"score": 3, "notes": "Learning objectives clearly stated and aligned to standards."},
                    "II-A": {"score": 4, "notes": "Excellent rapport with students. Respectful, warm interactions."},
                    "II-B": {"score": 3, "notes": "Positive culture for learning established. Students engaged and motivated."},
                    "III-A": {"score": 3, "notes": "Clear communication with students. Instructions were well understood."},
                    "III-B": {"score": 4, "notes": "Outstanding questioning techniques. Used wait time effectively, probing questions to deepen thinking."},
                    "III-C": {"score": 3, "notes": "Students actively engaged in learning activities. Good use of small groups."},
                    "IV-A": {"score": 3, "notes": "Teacher reflected thoughtfully on lesson effectiveness during post-observation conference."}
                }
            }
        },
        {
            "id": str(uuid4()),
            "teacher_name": "Michael Rodriguez", 
            "school_name": "Oak Middle School",
            "evaluator": "Assistant Principal Davis",
            "framework_id": "danielson_2023",
            "is_informal": True,
            "created_at": datetime.now(),
            "low_inference_notes": """
            Informal observation during math class. Teacher seemed overwhelmed and stressed. 
            Several students were off-task and talking. Teacher raised voice multiple times to 
            get attention. Lesson pace was very fast with little checking for understanding. 
            A few students appeared confused but didn't ask questions. Teacher mentioned being 
            tired and having too much on plate during brief conversation. Noticed decreased 
            enthusiasm compared to earlier observations this year.
            """,
            "evaluation": {
                "domains": {
                    "I-A": {"score": 2, "notes": "Content knowledge adequate but presentation unclear."},
                    "II-A": {"score": 2, "notes": "Some tension in interactions. Teacher seemed frustrated."},
                    "II-B": {"score": 1, "notes": "Learning environment disrupted by off-task behavior."},
                    "III-A": {"score": 2, "notes": "Communication unclear at times. Students seemed confused."},
                    "III-B": {"score": 1, "notes": "Minimal questioning. Mostly lecture format with little interaction."},
                    "III-C": {"score": 1, "notes": "Low student engagement. Many students off-task."},
                    "IV-A": {"score": 2, "notes": "Teacher acknowledged challenges but seemed overwhelmed to address them."}
                }
            }
        },
        {
            "id": str(uuid4()),
            "teacher_name": "Jennifer Chen",
            "school_name": "Washington High School", 
            "evaluator": "Department Head Wilson",
            "framework_id": "danielson_2023",
            "is_informal": False,
            "created_at": datetime.now(),
            "low_inference_notes": """
            Observed AP Biology class on cellular respiration. Teacher demonstrated exceptional 
            content expertise and pedagogical skills. Students were highly engaged in laboratory 
            investigation. Teacher moved seamlessly between groups providing differentiated support. 
            Excellent use of formative assessment throughout. Students asked thoughtful questions 
            showing deep understanding. Teacher reflected on lesson and identified areas for 
            improvement even though lesson was exemplary. Clear evidence of professional growth mindset.
            """,
            "evaluation": {
                "domains": {
                    "I-A": {"score": 4, "notes": "Exceptional content and pedagogical knowledge demonstrated."},
                    "I-B": {"score": 4, "notes": "Clear understanding of individual student needs and backgrounds."},
                    "I-C": {"score": 4, "notes": "Learning objectives clearly aligned with rigorous standards."},
                    "II-A": {"score": 4, "notes": "Outstanding relationships with students. Mutual respect evident."},
                    "II-B": {"score": 4, "notes": "Strong culture for learning. Students intrinsically motivated."},
                    "III-A": {"score": 4, "notes": "Crystal clear communication adapted to student needs."},
                    "III-B": {"score": 4, "notes": "Sophisticated questioning strategies promoting critical thinking."},
                    "III-C": {"score": 4, "notes": "Students deeply engaged in authentic learning experiences."},
                    "III-D": {"score": 4, "notes": "Seamless integration of formative assessment."},
                    "IV-A": {"score": 4, "notes": "Thoughtful reflection with specific plans for improvement."}
                }
            }
        }
    ]
    
    return sample_evaluations


def create_sample_org_config() -> Dict:
    """Create sample organization configuration."""
    
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
            "III-D": {"name": "Using Assessment in Instruction", "green": 3, "yellow": 2, "red": 1},
            "IV-A": {"name": "Reflecting on Teaching", "green": 3, "yellow": 2, "red": 1}
        },
        "global_thresholds": {
            "exemplar_teacher": 3.5,
            "proficient_teacher": 2.8,
            "developing_teacher": 2.0,
            "ineffective_teacher": 1.5
        },
        "risk_indicators": {
            "high_risk": {"red_domain_count": 2, "declining_trend_months": 6},
            "medium_risk": {"yellow_domain_count": 3, "stagnant_trend_months": 12}
        }
    }


async def test_basic_evaluation_processing():
    """Test basic evaluation processing functionality."""
    
    logger.info("Testing basic evaluation processing...")
    
    # Create LLM client (mock for testing)
    llm_client = create_llm_client("mock", delay_ms=50)
    
    # Create agent
    agent = DanielsonEvaluationAgent(llm_client=llm_client)
    
    # Get sample data
    sample_evaluations = create_sample_evaluation_data()
    org_config = create_sample_org_config()
    
    # Process each evaluation
    results = []
    for eval_data in sample_evaluations:
        # Convert to DanielsonEvaluation model
        evaluation = DanielsonEvaluation(**eval_data)
        
        # Create evaluation input
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config,
            analysis_focus="comprehensive",
            include_evidence=True,
            max_evidence_snippets=8
        )
        
        # Process evaluation
        result = await agent.execute_with_tracking(evaluation_input=eval_input)
        results.append(result)
        
        # Log results
        if result.success:
            eval_summary_data = result.data["evaluation_summary"]
            eval_summary = EvaluationSummary(**eval_summary_data)
            
            logger.info(f"Successfully processed evaluation for {eval_summary.teacher_name}")
            logger.info(f"  - Domains processed: {len(eval_summary.per_domain)}")
            logger.info(f"  - Evidence snippets: {len(eval_summary.evidence_snippets)}")
            logger.info(f"  - Key strengths: {len(eval_summary.key_strengths)}")
            logger.info(f"  - Key concerns: {len(eval_summary.key_concerns)}")
            logger.info(f"  - Flags: {eval_summary.flags}")
            
            # Show domain status breakdown
            domain_status = {}
            for domain_id, domain_summary in eval_summary.per_domain.items():
                status = domain_summary.status_color.value
                domain_status[status] = domain_status.get(status, 0) + 1
            logger.info(f"  - Domain status: {domain_status}")
            
        else:
            logger.error(f"Failed to process evaluation: {result.error}")
    
    return results


async def test_danielson_specific_agent():
    """Test Danielson-specific agent enhancements."""
    
    logger.info("Testing Danielson-specific agent...")
    
    # Create LLM client
    llm_client = create_llm_client("mock", delay_ms=75)
    
    # Create Danielson-specific agent
    agent = DanielsonSpecificAgent(llm_client=llm_client)
    
    # Get high-performing teacher sample
    sample_evaluations = create_sample_evaluation_data()
    high_performer = sample_evaluations[2]  # Jennifer Chen
    
    evaluation = DanielsonEvaluation(**high_performer)
    org_config = create_sample_org_config()
    
    eval_input = EvaluationInput(
        evaluation_data=evaluation,
        organization_config=org_config,
        analysis_focus="danielson_specific"
    )
    
    result = await agent.execute_with_tracking(evaluation_input=eval_input)
    
    if result.success:
        eval_summary_data = result.data["evaluation_summary"]
        eval_summary = EvaluationSummary(**eval_summary_data)
        
        logger.info(f"Danielson-specific analysis for {eval_summary.teacher_name}")
        
        # Test additional Danielson-specific methods
        domain_summaries = eval_summary.per_domain
        
        priority_score = await agent.calculate_danielson_priority_score(domain_summaries)
        logger.info(f"Priority score: {priority_score:.2f}")
        
        focus_domains = await agent.get_danielson_focus_domains(domain_summaries)
        logger.info(f"Focus domains: {focus_domains}")
        
        # Show enhanced analysis
        for domain_id, summary in domain_summaries.items():
            if hasattr(summary, 'priority_weight'):
                logger.info(f"Domain {domain_id}: score={summary.score}, weight={summary.priority_weight}")
    
    return result


async def test_error_handling():
    """Test error handling with malformed data."""
    
    logger.info("Testing error handling...")
    
    llm_client = create_llm_client("mock")
    agent = DanielsonEvaluationAgent(llm_client=llm_client)
    
    # Test with minimal/missing data
    minimal_eval = DanielsonEvaluation(
        id=uuid4(),
        teacher_name=None,
        school_name=None,
        evaluation=None,
        low_inference_notes=None,
        created_at=datetime.now()
    )
    
    org_config = create_sample_org_config()
    
    eval_input = EvaluationInput(
        evaluation_data=minimal_eval,
        organization_config=org_config
    )
    
    result = await agent.execute_with_tracking(evaluation_input=eval_input)
    
    if result.success:
        logger.info("Successfully handled minimal data")
        eval_summary_data = result.data["evaluation_summary"]
        eval_summary = EvaluationSummary(**eval_summary_data)
        logger.info(f"Teacher: {eval_summary.teacher_name}")
        logger.info(f"School: {eval_summary.school_name}")
        logger.info(f"Evidence count: {len(eval_summary.evidence_snippets)}")
    else:
        logger.error(f"Failed with minimal data: {result.error}")
    
    return result


async def test_batch_processing():
    """Test processing multiple evaluations in batch."""
    
    logger.info("Testing batch processing...")
    
    llm_client = create_llm_client("mock", delay_ms=30)
    agent = DanielsonEvaluationAgent(llm_client=llm_client)
    
    sample_evaluations = create_sample_evaluation_data()
    org_config = create_sample_org_config()
    
    # Process all evaluations concurrently
    tasks = []
    for eval_data in sample_evaluations:
        evaluation = DanielsonEvaluation(**eval_data)
        eval_input = EvaluationInput(
            evaluation_data=evaluation,
            organization_config=org_config
        )
        task = agent.execute_with_tracking(evaluation_input=eval_input)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Batch processed {len(results)} evaluations")
    
    # Aggregate results
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    
    logger.info(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    return results


async def demonstrate_output_formats():
    """Demonstrate different output formats and data structures."""
    
    logger.info("Demonstrating output formats...")
    
    llm_client = create_llm_client("mock")
    agent = DanielsonEvaluationAgent(llm_client=llm_client)
    
    # Process one evaluation
    sample_evaluations = create_sample_evaluation_data()
    evaluation = DanielsonEvaluation(**sample_evaluations[0])
    org_config = create_sample_org_config()
    
    eval_input = EvaluationInput(
        evaluation_data=evaluation,
        organization_config=org_config
    )
    
    result = await agent.execute_with_tracking(evaluation_input=eval_input)
    
    if result.success:
        # Show full JSON output
        eval_summary_data = result.data["evaluation_summary"]
        
        # Pretty print JSON
        logger.info("Full EvaluationSummary JSON:")
        print(json.dumps(eval_summary_data, indent=2, default=str))
        
        # Show individual components
        eval_summary = EvaluationSummary(**eval_summary_data)
        
        logger.info("\n=== EVALUATION SUMMARY ===")
        logger.info(f"Teacher: {eval_summary.teacher_name}")
        logger.info(f"School: {eval_summary.school_name}")
        logger.info(f"Date: {eval_summary.date}")
        logger.info(f"Type: {eval_summary.evaluation_type}")
        logger.info(f"Relevance: {eval_summary.relevance_to_question}")
        
        logger.info("\n=== DOMAIN STATUS ===")
        for domain_id, summary in eval_summary.per_domain.items():
            logger.info(f"{domain_id}: {summary.status_color.value} (score: {summary.score})")
        
        logger.info("\n=== FLAGS ===")
        for flag, value in eval_summary.flags.items():
            logger.info(f"{flag}: {value}")
        
        logger.info("\n=== KEY STRENGTHS ===")
        for strength in eval_summary.key_strengths:
            logger.info(f"- {strength}")
        
        logger.info("\n=== KEY CONCERNS ===")
        for concern in eval_summary.key_concerns:
            logger.info(f"- {concern}")
        
        logger.info("\n=== EVIDENCE SNIPPETS ===")
        for i, evidence in enumerate(eval_summary.evidence_snippets[:3], 1):
            logger.info(f"{i}. {evidence}")


async def main():
    """Main function to run all tests."""
    
    logger.info("Starting Danielson Evaluation Agent tests...")
    
    try:
        # Run test suite
        await test_basic_evaluation_processing()
        print("\n" + "="*60 + "\n")
        
        await test_danielson_specific_agent()
        print("\n" + "="*60 + "\n")
        
        await test_error_handling()
        print("\n" + "="*60 + "\n")
        
        await test_batch_processing()
        print("\n" + "="*60 + "\n")
        
        await demonstrate_output_formats()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())