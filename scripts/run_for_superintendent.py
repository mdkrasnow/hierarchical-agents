#!/usr/bin/env python3
"""
CLI script for superintendents to run hierarchical analysis.

This script:
1. Identifies the user as a superintendent
2. Loads evaluations across their entire district/organization
3. Runs full Evaluation → Teacher → School → District agent chain
4. Provides district-wide results and board communications
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from uuid import UUID

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import User, Organization, DanielsonEvaluation
from models.permissions import UserScope
from orchestration.hierarchical import HierarchicalOrchestrator, OrchestrationConfig
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Quiet down some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def load_user_and_organization(user_id: UUID) -> tuple[User, Organization]:
    """
    Load user and organization from database.
    
    In a real implementation, this would query Supabase.
    For now, we'll create mock objects for demonstration.
    """
    # TODO: Replace with actual database queries
    # This is a placeholder implementation
    
    user = User(
        id=user_id,
        email="superintendent@district.edu",
        role_id=UUID("12345678-1234-5678-9abc-123456789abc"),
        class_role="Superintendent",
        organization_id=UUID("87654321-4321-8765-cba9-987654321cba"),
        school_id=[],  # Superintendents have access to all schools
        created_at=datetime.now(),
        first_name="Dr. Robert",
        last_name="Johnson",
        active=True
    )
    
    organization = Organization(
        id=UUID("87654321-4321-8765-cba9-987654321cba"),
        name="Metro School District",
        performance_level_config={
            "domains": {
                "I-A": {"green": 3.0, "yellow": 2.0},
                "I-B": {"green": 3.0, "yellow": 2.0},
                # ... etc
            }
        },
        created_at=datetime.now(),
        type="district"
    )
    
    return user, organization


async def load_evaluations_for_superintendent(
    user_scope: UserScope,
    organization: Organization,
    months_back: int = 12
) -> List[DanielsonEvaluation]:
    """
    Load all evaluations in the district that the superintendent has access to.
    
    In a real implementation, this would query Supabase with proper filtering.
    """
    # TODO: Replace with actual database queries
    # This is a placeholder implementation
    
    print(f"📚 Loading district-wide evaluations")
    print(f"🏛️  Organization: {organization.name}")
    print(f"🗓️  Time period: Last {months_back} months")
    
    # Mock evaluations for demonstration
    evaluations = []
    
    # In real implementation:
    # 1. Query danielson table with organization_id filter
    # 2. Apply time range filter  
    # 3. Apply user permissions (superintendents see all)
    # 4. Return DanielsonEvaluation objects
    
    print(f"✅ Loaded {len(evaluations)} evaluations across district")
    return evaluations


def print_district_results(result):
    """Print district-focused results for superintendents."""
    print("\n" + "🌆 DISTRICT ANALYSIS RESULTS" + "\n" + "="*60)
    
    if not result.district_summary:
        print("❌ No district analysis results available")
        return
    
    district = result.district_summary
    
    print(f"\n🏛️  District: {district.organization_name}")
    print(f"🏫 Schools Analyzed: {district.num_schools_analyzed}")
    print(f"👥 Teachers Analyzed: {district.num_teachers_analyzed}")
    print(f"📈 Overall District Health: {district.overall_district_health.value.upper()}")
    print(f"⚠️  System Risk Level: {district.system_risk_level.value.upper()}")
    print(f"📊 Improvement Momentum: {district.improvement_momentum.value.upper()}")
    
    # Executive Summary
    if district.executive_summary:
        print(f"\n📋 Executive Summary:")
        print(f"   {district.executive_summary}")
    
    # Strategic Priorities
    if district.priority_domains:
        print(f"\n🎯 System-Wide Priority Domains:")
        for i, domain in enumerate(district.priority_domains, 1):
            print(f"   {i}. {domain}")
    
    if district.district_focus_areas:
        print(f"\n🔍 Strategic Focus Areas:")
        for i, area in enumerate(district.district_focus_areas, 1):
            print(f"   {i}. {area}")
    
    # District Strengths and Needs
    if district.district_strengths:
        print(f"\n✅ District Strengths:")
        for strength in district.district_strengths:
            print(f"   • {strength}")
    
    if district.district_needs:
        print(f"\n🎯 District Needs:")
        for need in district.district_needs:
            print(f"   • {need}")
    
    # School Performance
    if district.high_performing_schools:
        print(f"\n⭐ High-Performing Schools ({len(district.high_performing_schools)}):")
        for school in district.high_performing_schools:
            print(f"   • {school}")
    
    if district.schools_needing_support:
        print(f"\n🆘 Schools Needing Support ({len(district.schools_needing_support)}):")
        for school in district.schools_needing_support:
            print(f"   • {school}")
    
    # Board Stories
    if district.board_ready_stories:
        print(f"\n📊 Board Communication Stories:")
        for i, story in enumerate(district.board_ready_stories, 1):
            story_emoji = {"positive": "✅", "concern": "⚠️", "neutral": "📊"}.get(story.story_type, "📊")
            print(f"\n   {story_emoji} Story {i}: {story.title}")
            print(f"      {story.narrative}")
            if story.call_to_action:
                print(f"      🎯 Action: {story.call_to_action}")
    
    # Strategic Recommendations
    if district.recommended_PD_strategy:
        print(f"\n📚 Strategic PD Recommendations:")
        for i, rec in enumerate(district.recommended_PD_strategy, 1):
            print(f"   {i}. {rec}")
    
    if district.pilot_opportunities:
        print(f"\n🧪 Pilot Opportunities:")
        for i, pilot in enumerate(district.pilot_opportunities, 1):
            print(f"   {i}. {pilot}")
    
    # Resource Allocation
    if district.resource_allocation_priorities:
        print(f"\n💰 Resource Allocation Priorities:")
        for i, priority in enumerate(district.resource_allocation_priorities, 1):
            print(f"   {i}. {priority}")
    
    # Common PD Needs
    if district.common_PD_needs:
        print(f"\n👥 Common PD Needs Across District:")
        sorted_needs = sorted(district.common_PD_needs.items(), key=lambda x: x[1], reverse=True)
        for domain, teacher_count in sorted_needs[:5]:
            print(f"   • {domain}: {teacher_count} teachers")
    
    # Equity and Celebration
    if district.equity_concerns:
        print(f"\n⚖️  Equity Concerns:")
        for concern in district.equity_concerns:
            print(f"   • {concern}")
    
    if district.celebration_opportunities:
        print(f"\n🎉 Celebration Opportunities:")
        for celebration in district.celebration_opportunities:
            print(f"   • {celebration}")
    
    print("\n" + "="*60)


def print_school_comparison_table(result):
    """Print a comparison table of schools for superintendent overview."""
    if not result.school_summaries:
        return
    
    print(f"\n📊 SCHOOL PERFORMANCE OVERVIEW")
    print("="*80)
    
    # Header
    print(f"{'School Name':<25} {'Teachers':<8} {'Performance':<12} {'Risk':<8} {'Priority Domains':<15}")
    print("-"*80)
    
    # Sort schools by performance level and risk
    schools = sorted(result.school_summaries, 
                    key=lambda s: (s.overall_performance_level.value, s.school_risk_level.value))
    
    for school in schools:
        perf_emoji = {"green": "✅", "yellow": "⚠️", "red": "❌"}.get(school.overall_performance_level.value, "❓")
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(school.school_risk_level.value, "❓")
        
        priority_domains_str = ", ".join(school.priority_domains[:2]) if school.priority_domains else "None"
        if len(priority_domains_str) > 15:
            priority_domains_str = priority_domains_str[:12] + "..."
        
        print(f"{school.school_name[:24]:<25} "
              f"{school.num_teachers_analyzed:<8} "
              f"{perf_emoji} {school.overall_performance_level.value:<9} "
              f"{risk_emoji} {school.school_risk_level.value:<6} "
              f"{priority_domains_str:<15}")
    
    print("-"*80)


async def main():
    parser = argparse.ArgumentParser(
        description="Run hierarchical analysis for superintendents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --user-id 12345678-1234-5678-9abc-123456789abc
  %(prog)s --user-id 12345678-1234-5678-9abc-123456789abc --months-back 6 --verbose
  %(prog)s --user-id 12345678-1234-5678-9abc-123456789abc --board-focus --no-progress
        """
    )
    
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="Superintendent's user ID (UUID)"
    )
    
    parser.add_argument(
        "--months-back",
        type=int,
        default=12,
        help="How many months of evaluation history to analyze (default: 12)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=15,
        help="Maximum concurrent evaluation processing (default: 15)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output and debug logging"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators"
    )
    
    parser.add_argument(
        "--board-focus",
        action="store_true",
        help="Focus output on board communication stories and strategic items"
    )
    
    parser.add_argument(
        "--school-comparison",
        action="store_true",
        help="Include school-by-school comparison table"
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Parse user ID
        try:
            user_id = UUID(args.user_id)
        except ValueError:
            print(f"❌ Invalid user ID format: {args.user_id}")
            sys.exit(1)
        
        print(f"🚀 Starting superintendent analysis for user: {user_id}")
        
        # Load user and organization
        user, organization = await load_user_and_organization(user_id)
        
        # Create user scope
        user_scope = UserScope.from_user(user, organization)
        
        # Verify user is a superintendent
        if user_scope.role.value != "superintendent":
            print(f"❌ User role '{user_scope.role.value}' is not 'superintendent'")
            print("   This script is specifically for superintendents.")
            print("   Use 'run_for_principal.py' for principals.")
            sys.exit(1)
        
        print(f"✅ Verified superintendent access for: {user.first_name} {user.last_name}")
        print(f"🏛️  Organization: {organization.name}")
        
        # Load evaluations
        evaluations = await load_evaluations_for_superintendent(
            user_scope,
            organization,
            args.months_back
        )
        
        if not evaluations:
            print("⚠️  No evaluations found for analysis")
            print("   This could mean:")
            print("   • No evaluations in the specified time period")
            print("   • User doesn't have access to evaluation data")
            print("   • Database connection issues")
            sys.exit(0)
        
        # Configure orchestration
        config = OrchestrationConfig(
            max_concurrent_evaluations=args.max_concurrent,
            max_concurrent_teachers=8,
            max_concurrent_schools=4,
            max_history_months=args.months_back,
            verbose_output=args.verbose,
            show_progress=not args.no_progress,
            summary_first=True
        )
        
        # Initialize LLM client
        # TODO: Configure with actual API credentials
        llm_client = LLMClient()  # This would need proper initialization
        
        # Create orchestrator
        orchestrator = HierarchicalOrchestrator(
            llm_client=llm_client,
            config=config,
            logger_instance=logger
        )
        
        # Execute analysis
        result = await orchestrator.execute_for_user(
            user_id=user_id,
            user_scope=user_scope,
            organization=organization,
            evaluations=evaluations
        )
        
        if not result.success:
            print(f"❌ Analysis failed: {result.errors[0] if result.errors else 'Unknown error'}")
            sys.exit(1)
        
        # Display results based on focus
        if args.board_focus:
            # Show only board-relevant information
            if result.district_summary and result.district_summary.board_ready_stories:
                print("\n📊 BOARD COMMUNICATION PACKAGE")
                print("="*50)
                
                if result.district_summary.executive_summary:
                    print(f"\n📋 Executive Summary:")
                    print(f"   {result.district_summary.executive_summary}")
                
                for i, story in enumerate(result.district_summary.board_ready_stories, 1):
                    story_emoji = {"positive": "✅", "concern": "⚠️", "neutral": "📊"}.get(story.story_type, "📊")
                    print(f"\n{story_emoji} Board Story {i}: {story.title}")
                    print(f"   {story.narrative}")
                    if story.supporting_data:
                        print(f"   📊 Data: {story.supporting_data}")
                    if story.call_to_action:
                        print(f"   🎯 Action: {story.call_to_action}")
        else:
            # Full district analysis
            print_district_results(result)
            
            if args.school_comparison:
                print_school_comparison_table(result)
        
        # Save results if requested
        if args.save_results:
            import json
            with open(args.save_results, 'w') as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            print(f"💾 Results saved to: {args.save_results}")
        
        print(f"\n✅ Superintendent analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())