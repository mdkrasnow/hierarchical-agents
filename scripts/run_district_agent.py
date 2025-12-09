#!/usr/bin/env python3
"""
Script to run the DistrictAgent for processing school summaries into district-level analysis.

This script demonstrates the top-layer functionality of the hierarchical agent system,
consuming SchoolSummary objects and producing superintendent-level insights.

Usage:
    python scripts/run_district_agent.py --org-id <organization_id> [options]
    python scripts/run_district_agent.py --demo  # Run with demo data
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4, UUID

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.district import DistrictAgent, DistrictInput
from models import (
    SchoolSummary, 
    DomainStatus, 
    RiskLevel, 
    TrendDirection,
    PDCohort
)
from utils.llm import LLMClient, LLMProvider, LLMRequest, LLMResponse


class MockDistrictLLMProvider(LLMProvider):
    """Mock LLM provider for district agent demonstrations."""
    
    async def call_single(self, request: LLMRequest) -> LLMResponse:
        """Mock single call with realistic district responses."""
        if request.response_format:
            # Handle structured responses
            if "DistrictNarratives" in str(request.response_format):
                from agents.district import DistrictNarratives, BoardStory
                mock_data = DistrictNarratives(
                    district_strengths=[
                        "Strong instructional leadership capacity across multiple schools",
                        "Effective teacher collaboration and professional learning communities",
                        "Successful integration of data-driven decision making"
                    ],
                    district_needs=[
                        "System-wide focus on student engagement and motivation strategies", 
                        "Enhanced classroom management support for new teachers",
                        "Strengthened equity practices across all schools"
                    ],
                    executive_summary="The district demonstrates solid foundational strengths in leadership and collaboration, with clear opportunities for strategic improvement in student engagement and equitable teaching practices. Our data shows strong teacher performance in content knowledge while revealing system-wide needs in classroom management and student motivation strategies.",
                    recommended_pd_strategy=[
                        "Launch district-wide Student Engagement Initiative with research-based strategies",
                        "Implement cross-school Classroom Management Mentorship Program",
                        "Develop Equity and Culturally Responsive Teaching professional learning series",
                        "Create Teacher Leadership Academy for instructional coaches",
                        "Establish Data-Driven Instruction workshops for all schools"
                    ],
                    board_stories=[
                        BoardStory(
                            title="Celebrating Instructional Excellence",
                            narrative="Multiple schools demonstrate exceptional teacher performance with 60% of teachers showing strong content mastery and effective instructional practices.",
                            story_type="positive",
                            supporting_data={"exemplar_teachers": 15, "high_performing_schools": 2},
                            call_to_action=None
                        ),
                        BoardStory(
                            title="Strategic Professional Development Investment",
                            narrative="District-wide analysis reveals shared needs in student engagement, creating opportunities for efficient resource allocation and cross-school collaboration.",
                            story_type="neutral", 
                            supporting_data={"schools_with_shared_needs": 4, "teachers_to_benefit": 85},
                            call_to_action="Approve $150K investment in professional development initiatives"
                        ),
                        BoardStory(
                            title="Supporting Schools in Transition",
                            narrative="Two schools require intensive support in classroom management and teacher retention, representing 35% of our teaching force.",
                            story_type="concern",
                            supporting_data={"schools_needing_support": 2, "teachers_at_risk": 30},
                            call_to_action="Authorize additional coaching and mentorship resources"
                        )
                    ],
                    celebration_opportunities=[
                        "Recognize exemplary teachers at spring board meeting",
                        "Highlight successful cross-school collaboration at community forums",
                        "Feature innovative teaching practices in district newsletter"
                    ],
                    resource_priorities=[
                        "Student engagement professional development funding",
                        "Additional instructional coaching support for struggling schools", 
                        "Technology resources for data-driven instruction",
                        "Teacher wellness and retention program expansion"
                    ]
                )
                
                return LLMResponse(
                    content=mock_data.model_dump_json(),
                    parsed_data=mock_data,
                    latency_ms=250.0,
                    token_usage={"input_tokens": 800, "output_tokens": 400}
                )
        
        # Default mock response
        return LLMResponse(
            content="Mock LLM response for unstructured request",
            latency_ms=100.0,
            token_usage={"input_tokens": 50, "output_tokens": 25}
        )
    
    async def call_batch(self, requests: List[LLMRequest]):
        """Mock batch call."""
        import asyncio
        import time
        from utils.llm import BatchResult
        
        start_time = time.time()
        
        tasks = [self.call_single(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return BatchResult(
            responses=responses,
            failed_requests=[],
            total_latency_ms=total_latency_ms,
            successful_count=len(responses),
            failed_count=0
        )


def create_demo_school_summaries() -> List[SchoolSummary]:
    """Create demonstration school summaries with realistic data patterns."""
    
    # High-performing elementary school
    elementary_school = SchoolSummary(
        school_id=uuid4(),
        school_name="Riverside Elementary School",
        organization_id=uuid4(),
        analysis_period_start=datetime(2024, 9, 1, tzinfo=timezone.utc),
        analysis_period_end=datetime(2024, 12, 15, tzinfo=timezone.utc),
        num_teachers_analyzed=18,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 14, DomainStatus.YELLOW: 4, DomainStatus.RED: 0},
            "I-B": {DomainStatus.GREEN: 13, DomainStatus.YELLOW: 4, DomainStatus.RED: 1},
            "II-A": {DomainStatus.GREEN: 15, DomainStatus.YELLOW: 3, DomainStatus.RED: 0},
            "II-B": {DomainStatus.GREEN: 12, DomainStatus.YELLOW: 5, DomainStatus.RED: 1},
            "III-A": {DomainStatus.GREEN: 16, DomainStatus.YELLOW: 2, DomainStatus.RED: 0},
            "III-C": {DomainStatus.GREEN: 11, DomainStatus.YELLOW: 6, DomainStatus.RED: 1}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.78, DomainStatus.YELLOW: 0.22, DomainStatus.RED: 0.00},
            "I-B": {DomainStatus.GREEN: 0.72, DomainStatus.YELLOW: 0.22, DomainStatus.RED: 0.06},
            "II-A": {DomainStatus.GREEN: 0.83, DomainStatus.YELLOW: 0.17, DomainStatus.RED: 0.00},
            "II-B": {DomainStatus.GREEN: 0.67, DomainStatus.YELLOW: 0.28, DomainStatus.RED: 0.05},
            "III-A": {DomainStatus.GREEN: 0.89, DomainStatus.YELLOW: 0.11, DomainStatus.RED: 0.00},
            "III-C": {DomainStatus.GREEN: 0.61, DomainStatus.YELLOW: 0.33, DomainStatus.RED: 0.06}
        },
        PD_cohorts=[
            PDCohort(
                domain_id="III-C",
                focus_area="Student Engagement Strategies",
                teacher_names=["Ms. Johnson", "Mr. Davis", "Ms. Chen"],
                priority_level="medium",
                suggested_duration="4-5 sessions"
            )
        ],
        priority_domains=["III-C", "II-B"],
        school_strengths=[
            "Exceptional communication and instructional clarity",
            "Strong learning environment with positive classroom culture",
            "Excellent content knowledge and pedagogical skills"
        ],
        school_needs=[
            "Enhanced student engagement strategies for deeper learning",
            "Refined classroom management systems for optimal learning time"
        ],
        stories_for_principal=[
            "Leverage Ms. Rodriguez and Dr. Kim as mentor teachers for new hires",
            "Focus PLC time on student engagement research and strategies",
            "Celebrate strong school culture and communication practices"
        ],
        stories_for_supervisor_or_board=[
            "Riverside Elementary exemplifies instructional excellence with 78% of teachers demonstrating strong content mastery",
            "School demonstrates model practices in communication that could be shared district-wide"
        ],
        exemplar_teachers=["Ms. Rodriguez", "Dr. Kim", "Mr. Thompson", "Ms. Patel"],
        teachers_needing_support=["Ms. Williams"],
        overall_performance_level=DomainStatus.GREEN,
        school_risk_level=RiskLevel.LOW,
        improvement_trend=TrendDirection.IMPROVING
    )
    
    # Average-performing middle school
    middle_school = SchoolSummary(
        school_id=uuid4(),
        school_name="Lincoln Middle School",
        organization_id=uuid4(),
        analysis_period_start=datetime(2024, 9, 1, tzinfo=timezone.utc),
        analysis_period_end=datetime(2024, 12, 15, tzinfo=timezone.utc),
        num_teachers_analyzed=24,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 9, DomainStatus.YELLOW: 12, DomainStatus.RED: 3},
            "I-B": {DomainStatus.GREEN: 10, DomainStatus.YELLOW: 11, DomainStatus.RED: 3},
            "II-A": {DomainStatus.GREEN: 8, DomainStatus.YELLOW: 13, DomainStatus.RED: 3},
            "II-B": {DomainStatus.GREEN: 6, DomainStatus.YELLOW: 14, DomainStatus.RED: 4},
            "III-A": {DomainStatus.GREEN: 11, DomainStatus.YELLOW: 10, DomainStatus.RED: 3},
            "III-C": {DomainStatus.GREEN: 7, DomainStatus.YELLOW: 13, DomainStatus.RED: 4}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.38, DomainStatus.YELLOW: 0.50, DomainStatus.RED: 0.12},
            "I-B": {DomainStatus.GREEN: 0.42, DomainStatus.YELLOW: 0.46, DomainStatus.RED: 0.12},
            "II-A": {DomainStatus.GREEN: 0.33, DomainStatus.YELLOW: 0.54, DomainStatus.RED: 0.13},
            "II-B": {DomainStatus.GREEN: 0.25, DomainStatus.YELLOW: 0.58, DomainStatus.RED: 0.17},
            "III-A": {DomainStatus.GREEN: 0.46, DomainStatus.YELLOW: 0.42, DomainStatus.RED: 0.12},
            "III-C": {DomainStatus.GREEN: 0.29, DomainStatus.YELLOW: 0.54, DomainStatus.RED: 0.17}
        },
        PD_cohorts=[
            PDCohort(
                domain_id="II-B",
                focus_area="Classroom Management for Middle Grades",
                teacher_names=["Mr. Anderson", "Ms. Garcia", "Mr. Walsh", "Ms. Brooks"],
                priority_level="high",
                suggested_duration="6-8 sessions"
            ),
            PDCohort(
                domain_id="III-C",
                focus_area="Student Engagement & Motivation",
                teacher_names=["Mr. Lee", "Ms. Martinez", "Dr. Singh"],
                priority_level="high", 
                suggested_duration="5-6 sessions"
            )
        ],
        priority_domains=["II-B", "III-C", "I-A"],
        school_strengths=[
            "Committed teaching staff with strong collaborative culture",
            "Good foundation in instructional communication",
            "Emerging expertise in differentiated instruction"
        ],
        school_needs=[
            "Intensive classroom management support for student behavior",
            "Student engagement strategies for adolescent learners", 
            "Content knowledge development in priority subject areas"
        ],
        stories_for_principal=[
            "Implement structured classroom management PLC with external coaching support",
            "Partner with Riverside Elementary to observe engagement strategies",
            "Focus hiring on classroom management expertise for next year"
        ],
        stories_for_supervisor_or_board=[
            "Lincoln Middle School shows strong collaboration culture with clear needs in adolescent classroom management",
            "School would benefit from district-wide middle grades behavior support initiative"
        ],
        exemplar_teachers=["Ms. Foster", "Dr. Nguyen"],
        teachers_needing_support=["Mr. Anderson", "Ms. Garcia", "Mr. Walsh", "Ms. Brooks", "Mr. Lee"],
        overall_performance_level=DomainStatus.YELLOW,
        school_risk_level=RiskLevel.MEDIUM,
        improvement_trend=TrendDirection.STABLE
    )
    
    # Struggling high school needing intensive support
    high_school = SchoolSummary(
        school_id=uuid4(),
        school_name="Washington High School",
        organization_id=uuid4(),
        analysis_period_start=datetime(2024, 9, 1, tzinfo=timezone.utc),
        analysis_period_end=datetime(2024, 12, 15, tzinfo=timezone.utc),
        num_teachers_analyzed=32,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 6, DomainStatus.YELLOW: 14, DomainStatus.RED: 12},
            "I-B": {DomainStatus.GREEN: 8, DomainStatus.YELLOW: 13, DomainStatus.RED: 11},
            "II-A": {DomainStatus.GREEN: 5, DomainStatus.YELLOW: 15, DomainStatus.RED: 12},
            "II-B": {DomainStatus.GREEN: 4, DomainStatus.YELLOW: 12, DomainStatus.RED: 16},
            "III-A": {DomainStatus.GREEN: 9, DomainStatus.YELLOW: 12, DomainStatus.RED: 11},
            "III-C": {DomainStatus.GREEN: 3, DomainStatus.YELLOW: 13, DomainStatus.RED: 16}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.19, DomainStatus.YELLOW: 0.44, DomainStatus.RED: 0.37},
            "I-B": {DomainStatus.GREEN: 0.25, DomainStatus.YELLOW: 0.41, DomainStatus.RED: 0.34},
            "II-A": {DomainStatus.GREEN: 0.16, DomainStatus.YELLOW: 0.47, DomainStatus.RED: 0.37},
            "II-B": {DomainStatus.GREEN: 0.12, DomainStatus.YELLOW: 0.38, DomainStatus.RED: 0.50},
            "III-A": {DomainStatus.GREEN: 0.28, DomainStatus.YELLOW: 0.38, DomainStatus.RED: 0.34},
            "III-C": {DomainStatus.GREEN: 0.09, DomainStatus.YELLOW: 0.41, DomainStatus.RED: 0.50}
        },
        PD_cohorts=[
            PDCohort(
                domain_id="II-B",
                focus_area="Secondary Classroom Management Crisis Response",
                teacher_names=["Mr. Clark", "Ms. Turner", "Mr. Jackson", "Ms. Davis", "Dr. Wilson"],
                priority_level="high",
                suggested_duration="8-10 sessions"
            ),
            PDCohort(
                domain_id="III-C",
                focus_area="High School Student Engagement & Motivation",
                teacher_names=["Ms. Brown", "Mr. Miller", "Ms. Lopez", "Dr. Taylor"],
                priority_level="high",
                suggested_duration="6-8 sessions"
            ),
            PDCohort(
                domain_id="I-A", 
                focus_area="Content Knowledge & Secondary Pedagogy",
                teacher_names=["Mr. White", "Ms. Green", "Dr. Adams"],
                priority_level="high",
                suggested_duration="8-10 sessions"
            )
        ],
        priority_domains=["II-B", "III-C", "I-A", "II-A"],
        school_strengths=[
            "Dedicated teaching staff committed to student success despite challenges",
            "Strong department collaboration in core subject areas",
            "Emerging leadership capacity among veteran teachers"
        ],
        school_needs=[
            "Immediate intensive classroom management and behavior support",
            "Comprehensive student engagement and motivational strategies",
            "Content knowledge development and pedagogical training",
            "Teacher wellness and retention support systems"
        ],
        stories_for_principal=[
            "Request district crisis intervention team for classroom management support",
            "Implement emergency teacher mentoring program with successful neighboring schools",
            "Focus on teacher wellness initiatives to prevent further turnover",
            "Establish weekly coaching support for teachers in red domains"
        ],
        stories_for_supervisor_or_board=[
            "Washington High School requires immediate comprehensive intervention with 50% of teachers struggling in critical areas",
            "School represents significant retention risk with potential for 40% teacher turnover without intervention",
            "Despite challenges, school has committed staff and leadership foundation for recovery with proper support"
        ],
        exemplar_teachers=["Dr. Martinez", "Ms. Thompson"],
        teachers_needing_support=[
            "Mr. Clark", "Ms. Turner", "Mr. Jackson", "Ms. Davis", "Dr. Wilson",
            "Ms. Brown", "Mr. Miller", "Ms. Lopez", "Dr. Taylor", "Mr. White",
            "Ms. Green", "Dr. Adams", "Ms. Young", "Mr. Hall", "Ms. Scott"
        ],
        overall_performance_level=DomainStatus.RED,
        school_risk_level=RiskLevel.HIGH,
        improvement_trend=TrendDirection.DECLINING
    )
    
    # Smaller stable elementary school
    oak_elementary = SchoolSummary(
        school_id=uuid4(),
        school_name="Oak Elementary School", 
        organization_id=uuid4(),
        analysis_period_start=datetime(2024, 9, 1, tzinfo=timezone.utc),
        analysis_period_end=datetime(2024, 12, 15, tzinfo=timezone.utc),
        num_teachers_analyzed=12,
        domain_stats={
            "I-A": {DomainStatus.GREEN: 8, DomainStatus.YELLOW: 3, DomainStatus.RED: 1},
            "I-B": {DomainStatus.GREEN: 7, DomainStatus.YELLOW: 4, DomainStatus.RED: 1},
            "II-A": {DomainStatus.GREEN: 9, DomainStatus.YELLOW: 2, DomainStatus.RED: 1},
            "II-B": {DomainStatus.GREEN: 6, DomainStatus.YELLOW: 5, DomainStatus.RED: 1},
            "III-A": {DomainStatus.GREEN: 10, DomainStatus.YELLOW: 2, DomainStatus.RED: 0},
            "III-C": {DomainStatus.GREEN: 7, DomainStatus.YELLOW: 4, DomainStatus.RED: 1}
        },
        domain_percentages={
            "I-A": {DomainStatus.GREEN: 0.67, DomainStatus.YELLOW: 0.25, DomainStatus.RED: 0.08},
            "I-B": {DomainStatus.GREEN: 0.58, DomainStatus.YELLOW: 0.33, DomainStatus.RED: 0.09},
            "II-A": {DomainStatus.GREEN: 0.75, DomainStatus.YELLOW: 0.17, DomainStatus.RED: 0.08},
            "II-B": {DomainStatus.GREEN: 0.50, DomainStatus.YELLOW: 0.42, DomainStatus.RED: 0.08},
            "III-A": {DomainStatus.GREEN: 0.83, DomainStatus.YELLOW: 0.17, DomainStatus.RED: 0.00},
            "III-C": {DomainStatus.GREEN: 0.58, DomainStatus.YELLOW: 0.33, DomainStatus.RED: 0.09}
        },
        PD_cohorts=[
            PDCohort(
                domain_id="II-B",
                focus_area="Elementary Classroom Procedures",
                teacher_names=["Ms. Johnson", "Mr. Parker", "Ms. Riley"],
                priority_level="medium",
                suggested_duration="3-4 sessions"
            )
        ],
        priority_domains=["II-B", "III-C"],
        school_strengths=[
            "Excellent communication and instructional clarity across grade levels",
            "Strong school culture with positive learning environment",
            "Consistent performance in content knowledge and pedagogy"
        ],
        school_needs=[
            "Classroom management refinement for optimal learning time",
            "Student engagement enhancement for deeper learning experiences"
        ],
        stories_for_principal=[
            "Continue building on strong communication foundation",
            "Partner with Riverside Elementary to share engagement strategies",
            "Focus PLCs on procedural efficiency and student engagement"
        ],
        stories_for_supervisor_or_board=[
            "Oak Elementary demonstrates consistent solid performance with opportunities for targeted growth",
            "School represents stable foundation that could mentor other elementary schools"
        ],
        exemplar_teachers=["Ms. Anderson", "Dr. Roberts"],
        teachers_needing_support=["Ms. Johnson"],
        overall_performance_level=DomainStatus.GREEN,
        school_risk_level=RiskLevel.LOW,
        improvement_trend=TrendDirection.STABLE
    )
    
    return [elementary_school, middle_school, high_school, oak_elementary]


def load_school_summaries_from_file(file_path: str) -> List[SchoolSummary]:
    """Load school summaries from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        school_summaries = []
        for school_data in data:
            # Convert datetime strings back to datetime objects
            for date_field in ['analysis_period_start', 'analysis_period_end']:
                if school_data.get(date_field):
                    school_data[date_field] = datetime.fromisoformat(school_data[date_field])
            
            # Convert UUID strings back to UUID objects
            for uuid_field in ['school_id', 'organization_id']:
                if school_data.get(uuid_field):
                    school_data[uuid_field] = UUID(school_data[uuid_field])
            
            school_summary = SchoolSummary(**school_data)
            school_summaries.append(school_summary)
        
        return school_summaries
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []
    except Exception as e:
        print(f"Error loading school summaries: {e}")
        return []


async def run_district_agent(
    organization_id: Optional[UUID] = None,
    organization_name: str = "Sample School District",
    school_summaries: Optional[List[SchoolSummary]] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Run the DistrictAgent with provided or demo data.
    
    Args:
        organization_id: UUID of the organization (generated if not provided)
        organization_name: Name of the school district
        school_summaries: List of SchoolSummary objects (demo data if not provided)
        input_file: Path to JSON file containing school summaries
        output_file: Path to save district summary JSON output
        verbose: Enable detailed logging
        
    Returns:
        Dictionary containing district summary data
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load school summaries
        if input_file:
            logger.info(f"Loading school summaries from {input_file}")
            school_summaries = load_school_summaries_from_file(input_file)
            if not school_summaries:
                logger.error("Failed to load school summaries from file")
                return None
        elif not school_summaries:
            logger.info("Using demonstration school summaries")
            school_summaries = create_demo_school_summaries()
        
        # Generate organization ID if not provided
        if not organization_id:
            organization_id = uuid4()
        
        # Create district input
        district_input = DistrictInput(
            school_summaries=school_summaries,
            organization_id=organization_id,
            organization_name=organization_name,
            analysis_period_start=datetime(2024, 9, 1, tzinfo=timezone.utc),
            analysis_period_end=datetime(2024, 12, 15, tzinfo=timezone.utc),
            max_board_stories=6,
            min_school_count_for_ranking=3
        )
        
        logger.info(f"Processing district analysis for: {organization_name}")
        logger.info(f"Schools to analyze: {len(school_summaries)}")
        logger.info(f"Total teachers: {sum(s.num_teachers_analyzed for s in school_summaries)}")
        
        # Initialize agent with mock LLM client
        mock_provider = MockDistrictLLMProvider()
        llm_client = LLMClient(provider=mock_provider)
        district_agent = DistrictAgent(llm_client=llm_client)
        
        # Execute district analysis
        logger.info("Starting district-level analysis...")
        result = await district_agent.execute_with_tracking(district_input=district_input)
        
        if not result.success:
            logger.error(f"District analysis failed: {result.error}")
            return None
        
        # Extract district summary
        district_summary = result.data["district_summary"]
        
        # Display results
        print("\n" + "="*80)
        print(f"DISTRICT ANALYSIS RESULTS: {organization_name}")
        print("="*80)
        
        print(f"\nSCOPE:")
        print(f"  • Schools Analyzed: {district_summary['num_schools_analyzed']}")
        print(f"  • Teachers Analyzed: {district_summary['num_teachers_analyzed']}")
        print(f"  • Analysis Period: {district_input.analysis_period_start.strftime('%Y-%m-%d')} to {district_input.analysis_period_end.strftime('%Y-%m-%d')}")
        
        print(f"\nDISTRICT HEALTH METRICS:")
        print(f"  • Overall District Health: {district_summary['overall_district_health'].upper()}")
        print(f"  • System Risk Level: {district_summary['system_risk_level'].upper()}")
        print(f"  • Improvement Momentum: {district_summary['improvement_momentum'].upper()}")
        
        print(f"\nDISTRICT STRENGTHS:")
        for i, strength in enumerate(district_summary['district_strengths'], 1):
            print(f"  {i}. {strength}")
        
        print(f"\nDISTRICT NEEDS:")
        for i, need in enumerate(district_summary['district_needs'], 1):
            print(f"  {i}. {need}")
        
        print(f"\nSCHOOL CLASSIFICATIONS:")
        print(f"  • High-Performing Schools: {', '.join(district_summary['high_performing_schools']) if district_summary['high_performing_schools'] else 'None'}")
        print(f"  • Schools Needing Support: {', '.join(district_summary['schools_needing_support']) if district_summary['schools_needing_support'] else 'None'}")
        print(f"  • Pilot-Ready Schools: {', '.join(district_summary['pilot_opportunities']) if district_summary['pilot_opportunities'] else 'None'}")
        
        print(f"\nSTRATEGIC PD PRIORITIES:")
        for i, domain in enumerate(district_summary['priority_domains'][:5], 1):
            print(f"  {i}. {domain}")
        
        print(f"\nRECOMMENDED PD STRATEGY:")
        for i, strategy in enumerate(district_summary['recommended_PD_strategy'], 1):
            print(f"  {i}. {strategy}")
        
        print(f"\nBOARD-READY STORIES:")
        for i, story in enumerate(district_summary['board_ready_stories'], 1):
            print(f"  {i}. {story['title']} ({story['story_type']})")
            print(f"     {story['narrative']}")
            if story.get('call_to_action'):
                print(f"     Action: {story['call_to_action']}")
        
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"  {district_summary['executive_summary']}")
        
        print(f"\nCELEBRATION OPPORTUNITIES:")
        for i, opportunity in enumerate(district_summary['celebration_opportunities'], 1):
            print(f"  {i}. {opportunity}")
        
        print(f"\nRESOURCE ALLOCATION PRIORITIES:")
        for i, priority in enumerate(district_summary['resource_allocation_priorities'], 1):
            print(f"  {i}. {priority}")
        
        if district_summary.get('equity_concerns'):
            print(f"\nEQUITY CONCERNS:")
            for i, concern in enumerate(district_summary['equity_concerns'], 1):
                print(f"  {i}. {concern}")
        
        # Display cross-school rankings if available
        if district_summary.get('school_rankings_by_domain'):
            print(f"\nSCHOOL RANKINGS (Overall Performance):")
            overall_rankings = district_summary['school_rankings_by_domain'].get('overall', [])
            for ranking in overall_rankings:
                print(f"  {ranking['overall_rank']}. {ranking['school_name']}")
                if ranking.get('standout_areas'):
                    print(f"     Strengths: {', '.join(ranking['standout_areas'])}")
                if ranking.get('improvement_areas'):
                    print(f"     Growth Areas: {', '.join(ranking['improvement_areas'])}")
        
        # Save to file if requested
        if output_file:
            try:
                # Convert datetime and UUID objects to strings for JSON serialization
                serializable_summary = json.loads(json.dumps(district_summary, default=str))
                
                with open(output_file, 'w') as f:
                    json.dump(serializable_summary, f, indent=2)
                logger.info(f"District summary saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save output file: {e}")
        
        logger.info("District analysis completed successfully")
        return district_summary
        
    except Exception as e:
        logger.error(f"District analysis failed with exception: {e}", exc_info=True)
        return None


def main():
    """Main CLI interface for running district agent."""
    parser = argparse.ArgumentParser(
        description="Run DistrictAgent for superintendent-level insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with demo data
  python scripts/run_district_agent.py --demo
  
  # Run with custom organization
  python scripts/run_district_agent.py --org-id "550e8400-e29b-41d4-a716-446655440000" --org-name "Riverside School District"
  
  # Load from file and save results
  python scripts/run_district_agent.py --input-file school_summaries.json --output-file district_results.json
        """
    )
    
    # Input options
    parser.add_argument(
        "--org-id",
        type=str,
        help="Organization UUID (generated if not provided)"
    )
    parser.add_argument(
        "--org-name", 
        type=str,
        default="Sample School District",
        help="Organization name (default: Sample School District)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="JSON file containing school summaries"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with demonstration data"
    )
    
    # Output options
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save district summary to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Convert org_id string to UUID if provided
    org_id = None
    if args.org_id:
        try:
            org_id = UUID(args.org_id)
        except ValueError:
            print(f"Error: Invalid UUID format for org-id: {args.org_id}")
            return 1
    
    # Run district agent
    result = asyncio.run(run_district_agent(
        organization_id=org_id,
        organization_name=args.org_name,
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose
    ))
    
    if result is None:
        print("District analysis failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())