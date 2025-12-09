#!/usr/bin/env python3
"""
Script to run TeacherAgent for aggregating evaluation summaries.

Usage:
    python scripts/run_teacher_agent.py --teacher-name "Jane Smith" --evaluations evaluation1.json,evaluation2.json
    python scripts/run_teacher_agent.py --input teacher_input.json --output teacher_summary.json
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.teacher import TeacherAgent, TeacherInput
from models import EvaluationSummary
from utils.llm import create_llm_client


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('teacher_agent.log')
        ]
    )


def load_evaluation_summaries(file_paths: List[str]) -> List[EvaluationSummary]:
    """Load evaluation summaries from JSON files."""
    summaries = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Handle both single summary and list of summaries
            if isinstance(data, list):
                for item in data:
                    summaries.append(EvaluationSummary.model_validate(item))
            else:
                summaries.append(EvaluationSummary.model_validate(data))
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            sys.exit(1)
    
    return summaries


def create_sample_evaluations(teacher_name: str) -> List[EvaluationSummary]:
    """Create sample evaluation summaries for testing."""
    from models import DomainSummary, DomainStatus
    
    # Sample domain summaries
    domains_eval1 = {
        "I-A": DomainSummary(
            domain_id="I-A",
            score=3.2,
            status_color=DomainStatus.GREEN,
            summary="Strong content knowledge demonstrated",
            growth_signals=["Uses content expertise effectively"],
            concern_signals=[],
            evidence_quotes=["Teacher demonstrated deep understanding of subject matter"]
        ),
        "II-B": DomainSummary(
            domain_id="II-B",
            score=2.1,
            status_color=DomainStatus.YELLOW,
            summary="Classroom management needs development",
            growth_signals=[],
            concern_signals=["Inconsistent behavior management"],
            evidence_quotes=["Some disruptions noted during instruction"]
        ),
        "III-C": DomainSummary(
            domain_id="III-C",
            score=3.5,
            status_color=DomainStatus.GREEN,
            summary="Excellent student engagement",
            growth_signals=["Students highly engaged", "Interactive teaching methods"],
            concern_signals=[],
            evidence_quotes=["All students actively participating in discussions"]
        )
    }
    
    domains_eval2 = {
        "I-A": DomainSummary(
            domain_id="I-A",
            score=3.0,
            status_color=DomainStatus.GREEN,
            summary="Consistent content delivery",
            growth_signals=["Clear explanations"],
            concern_signals=[],
            evidence_quotes=["Students understand complex concepts"]
        ),
        "II-B": DomainSummary(
            domain_id="II-B",
            score=2.3,
            status_color=DomainStatus.YELLOW,
            summary="Slight improvement in management",
            growth_signals=["Better routine establishment"],
            concern_signals=["Still some disruptions"],
            evidence_quotes=["Improved transition times observed"]
        ),
        "III-C": DomainSummary(
            domain_id="III-C",
            score=3.3,
            status_color=DomainStatus.GREEN,
            summary="Continues strong engagement",
            growth_signals=["Variety in activities"],
            concern_signals=[],
            evidence_quotes=["Students eager to participate"]
        )
    }
    
    eval1 = EvaluationSummary(
        teacher_id=uuid4(),
        teacher_name=teacher_name,
        school_id=uuid4(),
        school_name="Sample Elementary School",
        evaluation_id=uuid4(),
        date=datetime(2024, 9, 15),
        per_domain=domains_eval1,
        flags={
            "needs_PD": ["II-B"],
            "exemplar": False,
            "risk_of_leaving": False,
            "burnout_signals": False
        },
        evidence_snippets=[
            "Teacher demonstrates strong pedagogical content knowledge",
            "Students respond positively to interactive methods",
            "Classroom transitions could be smoother"
        ],
        key_strengths=[
            "Strong subject matter expertise",
            "Excellent student rapport"
        ],
        key_concerns=[
            "Classroom management consistency"
        ],
        relevance_to_question="high",
        evaluation_type="formal"
    )
    
    eval2 = EvaluationSummary(
        teacher_id=uuid4(),
        teacher_name=teacher_name,
        school_id=uuid4(),
        school_name="Sample Elementary School",
        evaluation_id=uuid4(),
        date=datetime(2024, 11, 20),
        per_domain=domains_eval2,
        flags={
            "needs_PD": ["II-B"],
            "exemplar": False,
            "risk_of_leaving": False,
            "burnout_signals": False
        },
        evidence_snippets=[
            "Continued growth in content delivery",
            "Improvement noted in classroom routines",
            "Student engagement remains high"
        ],
        key_strengths=[
            "Consistent content expertise",
            "Strong student relationships"
        ],
        key_concerns=[
            "Ongoing management challenges"
        ],
        relevance_to_question="high",
        evaluation_type="formal"
    )
    
    return [eval1, eval2]


async def main():
    parser = argparse.ArgumentParser(description="Run TeacherAgent for evaluation aggregation")
    
    # Input options
    parser.add_argument("--input", help="JSON file with TeacherInput data")
    parser.add_argument("--teacher-name", help="Name of teacher to analyze")
    parser.add_argument("--evaluations", help="Comma-separated list of evaluation summary JSON files")
    parser.add_argument("--sample", action="store_true", help="Use sample data for testing")
    
    # Output options
    parser.add_argument("--output", help="Output file for teacher summary (default: stdout)")
    
    # Configuration
    parser.add_argument("--llm-provider", choices=["mock", "claude"], default="mock",
                       help="LLM provider to use")
    parser.add_argument("--api-key", help="API key for LLM provider (if required)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--pd-focus-limit", type=int, default=3, 
                       help="Maximum number of PD focus areas")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create LLM client
        llm_client = create_llm_client(
            provider_type=args.llm_provider,
            api_key=args.api_key
        )
        
        # Create TeacherAgent
        agent = TeacherAgent(llm_client=llm_client)
        
        # Prepare input data
        if args.input:
            # Load from input file
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            teacher_input = TeacherInput.model_validate(input_data)
            
        elif args.sample:
            # Use sample data
            teacher_name = args.teacher_name or "Sample Teacher"
            evaluations = create_sample_evaluations(teacher_name)
            teacher_input = TeacherInput(
                evaluations=evaluations,
                teacher_name=teacher_name,
                analysis_period_start=datetime(2024, 9, 1),
                analysis_period_end=datetime(2024, 11, 30),
                pd_focus_limit=args.pd_focus_limit
            )
            
        elif args.teacher_name and args.evaluations:
            # Load from individual files
            evaluation_files = [f.strip() for f in args.evaluations.split(',')]
            evaluations = load_evaluation_summaries(evaluation_files)
            
            teacher_input = TeacherInput(
                evaluations=evaluations,
                teacher_name=args.teacher_name,
                pd_focus_limit=args.pd_focus_limit
            )
            
        else:
            logger.error("Must provide either --input, --sample, or --teacher-name with --evaluations")
            parser.print_help()
            sys.exit(1)
        
        # Run analysis
        logger.info(f"Starting teacher analysis for {teacher_input.teacher_name}")
        logger.info(f"Processing {len(teacher_input.evaluations)} evaluations")
        
        result = await agent.execute_with_tracking(teacher_input=teacher_input)
        
        if result.success:
            teacher_summary = result.data["teacher_summary"]
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(teacher_summary, f, indent=2, default=str)
                logger.info(f"Results written to {args.output}")
            else:
                print(json.dumps(teacher_summary, indent=2, default=str))
            
            # Log summary statistics
            logger.info("Analysis completed successfully")
            logger.info(f"Risk level: {teacher_summary['risk_level']}")
            logger.info(f"PD recommendations: {len(teacher_summary['recommended_PD_focus'])}")
            logger.info(f"Is exemplar: {teacher_summary['is_exemplar']}")
            
        else:
            logger.error(f"Analysis failed: {result.error}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())