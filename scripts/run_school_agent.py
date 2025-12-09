#!/usr/bin/env python3
"""
Script to run SchoolAgent analysis for a specific school.

This script demonstrates the third layer of the hierarchical agent system,
processing TeacherSummary objects to produce SchoolSummary with domain 
statistics, PD cohorts, and narrative summaries.

Usage:
    python scripts/run_school_agent.py --school-id <uuid> [--school-name <name>]
    python scripts/run_school_agent.py --demo  # Run with mock data
"""

import argparse
import asyncio
import json
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.school import SchoolAgent, SchoolInput
from models import TeacherSummary, DomainSummary, DomainStatus, RiskLevel, TrendDirection
from utils.llm import LLMClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_mock_teacher_summaries(school_name: str) -> list[TeacherSummary]:
    """Create mock teacher summaries for demonstration purposes."""
    
    # Mock teacher 1: High performer / Exemplar
    teacher1 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Sarah Mitchell",
        school_name=school_name,
        num_evaluations=4,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=3.8,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Exceptional content knowledge and pedagogical skills",
                growth_signals=["Uses research-based strategies", "Adapts to student needs"],
                evidence_quotes=["Demonstrates mastery of subject matter", "Effective differentiation"]
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=3.6,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.IMPROVING,
                summary="Strong classroom management with positive culture",
                growth_signals=["Clear expectations", "Student-centered environment"],
                evidence_quotes=["Students engaged and on-task", "Respectful classroom community"]
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=3.9,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Outstanding student engagement strategies",
                growth_signals=["Active learning techniques", "Student voice encouraged"],
                evidence_quotes=["Students actively participating", "High level of engagement"]
            ),
            "IV-A": DomainSummary(
                domain_id="IV-A",
                score=3.7,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.IMPROVING,
                summary="Strong professional reflection and growth mindset",
                growth_signals=["Seeks feedback", "Self-directed learning"],
                evidence_quotes=["Reflects on practice", "Implements suggestions"]
            )
        },
        recommended_PD_domains=[],
        recommended_PD_focus=[],
        risk_level=RiskLevel.LOW,
        overall_performance_trend=TrendDirection.IMPROVING,
        is_exemplar=True,
        needs_immediate_support=False,
        domain_distribution={
            DomainStatus.GREEN: 4,
            DomainStatus.YELLOW: 0,
            DomainStatus.RED: 0
        },
        overall_short_summary="Exemplary teacher demonstrating best practices across all domains"
    )
    
    # Mock teacher 2: Solid performer with some growth areas
    teacher2 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Marcus Rodriguez",
        school_name=school_name,
        num_evaluations=3,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=3.2,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Good content knowledge with room for pedagogical growth"
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=2.8,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.IMPROVING,
                summary="Developing classroom management skills"
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=2.9,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Student engagement strategies emerging"
            ),
            "IV-A": DomainSummary(
                domain_id="IV-A",
                score=3.1,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Professional and reflective practitioner"
            )
        },
        recommended_PD_domains=["II-B", "III-C"],
        recommended_PD_focus=["Classroom Management", "Student Engagement"],
        risk_level=RiskLevel.LOW,
        overall_performance_trend=TrendDirection.IMPROVING,
        is_exemplar=False,
        needs_immediate_support=False,
        domain_distribution={
            DomainStatus.GREEN: 2,
            DomainStatus.YELLOW: 2,
            DomainStatus.RED: 0
        },
        overall_short_summary="Solid teacher showing growth in classroom management and engagement"
    )
    
    # Mock teacher 3: Struggling teacher needing support
    teacher3 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Jennifer Park",
        school_name=school_name,
        num_evaluations=2,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=1.9,
                status_color=DomainStatus.RED,
                trend=TrendDirection.DECLINING,
                summary="Content knowledge gaps affecting instruction",
                concern_signals=["Limited subject expertise", "Relies heavily on textbook"]
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=2.1,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Classroom management challenges",
                concern_signals=["Student disruptions", "Unclear procedures"]
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=2.3,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Limited engagement strategies",
                concern_signals=["Passive learning", "Low participation"]
            ),
            "IV-A": DomainSummary(
                domain_id="IV-A",
                score=2.5,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Beginning to reflect on practice"
            )
        },
        recommended_PD_domains=["I-A", "II-B", "III-C"],
        recommended_PD_focus=["Content Knowledge and Pedagogy", "Classroom Management", "Student Engagement"],
        risk_level=RiskLevel.HIGH,
        overall_performance_trend=TrendDirection.DECLINING,
        is_exemplar=False,
        needs_immediate_support=True,
        domain_distribution={
            DomainStatus.GREEN: 0,
            DomainStatus.YELLOW: 3,
            DomainStatus.RED: 1
        },
        overall_short_summary="First-year teacher requiring intensive support and mentoring"
    )
    
    # Mock teacher 4: Veteran teacher, mostly solid
    teacher4 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="David Thompson",
        school_name=school_name,
        num_evaluations=3,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=3.4,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Strong content knowledge and experience"
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=3.3,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Well-established classroom management"
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=2.7,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.STABLE,
                summary="Traditional engagement methods, could modernize"
            ),
            "IV-A": DomainSummary(
                domain_id="IV-A",
                score=3.0,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.STABLE,
                summary="Professional and experienced"
            )
        },
        recommended_PD_domains=["III-C"],
        recommended_PD_focus=["Student Engagement"],
        risk_level=RiskLevel.LOW,
        overall_performance_trend=TrendDirection.STABLE,
        is_exemplar=False,
        needs_immediate_support=False,
        domain_distribution={
            DomainStatus.GREEN: 3,
            DomainStatus.YELLOW: 1,
            DomainStatus.RED: 0
        },
        overall_short_summary="Experienced teacher with opportunity to enhance engagement strategies"
    )
    
    # Mock teacher 5: Another developing teacher
    teacher5 = TeacherSummary(
        teacher_id=uuid4(),
        teacher_name="Lisa Chen",
        school_name=school_name,
        num_evaluations=3,
        per_domain_overview={
            "I-A": DomainSummary(
                domain_id="I-A",
                score=2.9,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.IMPROVING,
                summary="Growing content knowledge"
            ),
            "II-B": DomainSummary(
                domain_id="II-B",
                score=2.6,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.IMPROVING,
                summary="Classroom management improving with support"
            ),
            "III-C": DomainSummary(
                domain_id="III-C",
                score=3.1,
                status_color=DomainStatus.GREEN,
                trend=TrendDirection.IMPROVING,
                summary="Natural ability to engage students"
            ),
            "IV-A": DomainSummary(
                domain_id="IV-A",
                score=2.8,
                status_color=DomainStatus.YELLOW,
                trend=TrendDirection.IMPROVING,
                summary="Developing reflection skills"
            )
        },
        recommended_PD_domains=["II-B"],
        recommended_PD_focus=["Classroom Management"],
        risk_level=RiskLevel.LOW,
        overall_performance_trend=TrendDirection.IMPROVING,
        is_exemplar=False,
        needs_immediate_support=False,
        domain_distribution={
            DomainStatus.GREEN: 1,
            DomainStatus.YELLOW: 3,
            DomainStatus.RED: 0
        },
        overall_short_summary="Developing teacher with strong engagement skills and positive trajectory"
    )
    
    return [teacher1, teacher2, teacher3, teacher4, teacher5]


async def run_school_analysis(school_id: str, school_name: str, use_mock_data: bool = False) -> None:
    """Run school analysis for the specified school."""
    
    logger.info(f"Starting school analysis for: {school_name} (ID: {school_id})")
    
    # Initialize LLM client (mock for this example)
    llm_client = None  # Would normally initialize real LLM client
    
    # Create SchoolAgent
    school_agent = SchoolAgent(llm_client=llm_client)
    
    if use_mock_data:
        # Use mock teacher summaries for demonstration
        teacher_summaries = create_mock_teacher_summaries(school_name)
        logger.info(f"Using mock data with {len(teacher_summaries)} teachers")
    else:
        # In real implementation, would fetch from database
        logger.warning("Real data fetching not implemented - using mock data")
        teacher_summaries = create_mock_teacher_summaries(school_name)
    
    # Create school input
    school_input = SchoolInput(
        teacher_summaries=teacher_summaries,
        school_id=UUID(school_id) if school_id != "demo" else uuid4(),
        school_name=school_name,
        analysis_period_start=datetime.now() - timedelta(days=180),
        analysis_period_end=datetime.now(),
        max_cohort_size=8,
        min_cohort_size=3
    )
    
    # Execute school analysis
    logger.info("Executing school analysis...")
    result = await school_agent.execute(school_input)
    
    if not result.success:
        logger.error(f"School analysis failed: {result.error}")
        return
    
    # Extract and display results
    school_summary = result.data["school_summary"]
    
    print("\n" + "="*60)
    print(f"SCHOOL ANALYSIS RESULTS: {school_name}")
    print("="*60)
    
    print(f"\nBasic Information:")
    print(f"  Teachers Analyzed: {school_summary['num_teachers_analyzed']}")
    print(f"  Overall Performance: {school_summary['overall_performance_level']}")
    print(f"  School Risk Level: {school_summary['school_risk_level']}")
    print(f"  Improvement Trend: {school_summary['improvement_trend']}")
    
    print(f"\nDomain Statistics:")
    for domain_id, stats in school_summary['domain_stats'].items():
        percentages = school_summary['domain_percentages'][domain_id]
        print(f"  {domain_id}:")
        print(f"    Green: {stats['green']} ({percentages['green']:.1%})")
        print(f"    Yellow: {stats['yellow']} ({percentages['yellow']:.1%})")
        print(f"    Red: {stats['red']} ({percentages['red']:.1%})")
    
    print(f"\nProfessional Development Cohorts:")
    if school_summary['PD_cohorts']:
        for cohort in school_summary['PD_cohorts']:
            print(f"  {cohort['focus_area']} (Priority: {cohort['priority_level']})")
            print(f"    Teachers: {', '.join(cohort['teacher_names'])}")
            print(f"    Duration: {cohort['suggested_duration']}")
            print()
    else:
        print("  No PD cohorts identified (insufficient teachers per domain)")
    
    print(f"\nPriority Domains for School-wide Focus:")
    for i, domain in enumerate(school_summary['priority_domains'], 1):
        print(f"  {i}. {domain}")
    
    print(f"\nTeacher Recognition & Support:")
    print(f"  Exemplar Teachers: {', '.join(school_summary['exemplar_teachers']) if school_summary['exemplar_teachers'] else 'None identified'}")
    print(f"  Teachers Needing Support: {', '.join(school_summary['teachers_needing_support']) if school_summary['teachers_needing_support'] else 'None identified'}")
    
    print(f"\nSchool Strengths:")
    for strength in school_summary['school_strengths']:
        print(f"  • {strength}")
    
    print(f"\nSchool Needs:")
    for need in school_summary['school_needs']:
        print(f"  • {need}")
    
    print(f"\nStories for Principal:")
    for story in school_summary['stories_for_principal']:
        print(f"  • {story}")
    
    print(f"\nStories for Supervisor/Board:")
    for story in school_summary['stories_for_supervisor_or_board']:
        print(f"  • {story}")
    
    print("\n" + "="*60)
    
    # Optionally save to file
    output_file = Path(f"school_analysis_{school_id}.json")
    with open(output_file, 'w') as f:
        json.dump(school_summary, f, indent=2, default=str)
    
    logger.info(f"Analysis complete. Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SchoolAgent analysis")
    parser.add_argument("--school-id", type=str, help="School UUID to analyze")
    parser.add_argument("--school-name", type=str, help="School name")
    parser.add_argument("--demo", action="store_true", help="Run demo with mock data")
    
    args = parser.parse_args()
    
    if args.demo:
        school_id = "demo"
        school_name = "Lincoln Elementary School"
    elif args.school_id:
        school_id = args.school_id
        school_name = args.school_name or f"School {school_id}"
    else:
        parser.error("Either --school-id or --demo is required")
    
    # Run the analysis
    asyncio.run(run_school_analysis(school_id, school_name, use_mock_data=True))


if __name__ == "__main__":
    main()