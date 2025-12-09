#!/usr/bin/env python3
"""
CLI script for principals to run hierarchical analysis.

This script:
1. Identifies the user as a principal
2. Loads evaluations for their school(s)
3. Runs Evaluation → Teacher → School agent chain
4. Provides school-focused results and PD planning
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
        email="principal@school.edu",
        role_id=UUID("12345678-1234-5678-9abc-123456789abc"),
        class_role="Principal",
        organization_id=UUID("87654321-4321-8765-cba9-987654321cba"),
        school_id=[UUID("11111111-1111-1111-1111-111111111111")],
        created_at=datetime.now(),
        first_name="Jane",
        last_name="Smith",
        active=True
    )
    
    organization = Organization(
        id=UUID("87654321-4321-8765-cba9-987654321cba"),
        name="Example School District",
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


async def load_evaluations_for_principal(
    user_scope: UserScope,
    organization: Organization,
    months_back: int = 12
) -> List[DanielsonEvaluation]:
    """
    Load evaluations that the principal has access to.
    
    In a real implementation, this would query Supabase with proper filtering.
    """
    # TODO: Replace with actual database queries
    # This is a placeholder implementation
    
    print(f"📚 Loading evaluations for school(s): {user_scope.school_names or 'All assigned schools'}")
    print(f"🗓️  Time period: Last {months_back} months")
    
    # Mock evaluations for demonstration
    evaluations = []
    
    # In real implementation:
    # 1. Query danielson table with school_id filter
    # 2. Apply time range filter
    # 3. Apply user permissions
    # 4. Return DanielsonEvaluation objects
    
    print(f"✅ Loaded {len(evaluations)} evaluations")
    return evaluations


def print_school_results(result):
    """Print school-focused results for principals."""
    print("\n" + "🏫 SCHOOL ANALYSIS RESULTS" + "\n" + "="*50)
    
    if not result.school_summaries:
        print("❌ No school analysis results available")
        return
    
    for school in result.school_summaries:
        print(f"\n🏛️  School: {school.school_name}")
        print(f"👥 Teachers Analyzed: {school.num_teachers_analyzed}")
        print(f"📈 Overall Performance: {school.overall_performance_level.value.upper()}")
        print(f"⚠️  Risk Level: {school.school_risk_level.value.upper()}")
        
        # Domain priorities
        if school.priority_domains:
            print(f"\n🎯 Priority Development Areas:")
            for i, domain in enumerate(school.priority_domains[:3], 1):
                print(f"   {i}. {domain}")
        
        # Teacher insights
        if school.exemplar_teachers:
            print(f"\n⭐ Exemplar Teachers ({len(school.exemplar_teachers)}):")
            for teacher in school.exemplar_teachers[:5]:
                print(f"   • {teacher}")
        
        if school.teachers_needing_support:
            print(f"\n🆘 Teachers Needing Support ({len(school.teachers_needing_support)}):")
            for teacher in school.teachers_needing_support[:5]:
                print(f"   • {teacher}")
        
        # PD Cohorts
        if school.PD_cohorts:
            print(f"\n👥 Professional Development Cohorts:")
            for cohort in school.PD_cohorts[:3]:
                print(f"   📚 {cohort.focus_area} ({len(cohort.teacher_names)} teachers)")
                print(f"      Priority: {cohort.priority_level}, Duration: {cohort.suggested_duration}")
        
        # Principal stories
        if school.stories_for_principal:
            print(f"\n📖 Principal Action Items:")
            for i, story in enumerate(school.stories_for_principal, 1):
                print(f"   {i}. {story}")
        
        print("\n" + "-"*50)


async def main():
    parser = argparse.ArgumentParser(
        description="Run hierarchical analysis for principals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --user-id 12345678-1234-5678-9abc-123456789abc
  %(prog)s --user-id 12345678-1234-5678-9abc-123456789abc --months-back 6 --verbose
  %(prog)s --user-id 12345678-1234-5678-9abc-123456789abc --no-progress
        """
    )
    
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="Principal's user ID (UUID)"
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
        default=10,
        help="Maximum concurrent evaluation processing (default: 10)"
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
        "--summary-only",
        action="store_true",
        help="Show only high-level summary, skip detailed results"
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
        
        print(f"🚀 Starting principal analysis for user: {user_id}")
        
        # Load user and organization
        user, organization = await load_user_and_organization(user_id)
        
        # Create user scope
        user_scope = UserScope.from_user(user, organization)
        
        # Verify user is a principal
        if user_scope.role.value != "principal":
            print(f"❌ User role '{user_scope.role.value}' is not 'principal'")
            print("   This script is specifically for principals.")
            print("   Use 'run_for_superintendent.py' for superintendents.")
            sys.exit(1)
        
        print(f"✅ Verified principal access for: {user.first_name} {user.last_name}")
        print(f"🏛️  Organization: {organization.name}")
        
        # Load evaluations
        evaluations = await load_evaluations_for_principal(
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
            max_concurrent_teachers=5,
            max_concurrent_schools=2,
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
        
        # Display results
        if not args.summary_only:
            print_school_results(result)
        
        # Save results if requested
        if args.save_results:
            import json
            with open(args.save_results, 'w') as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            print(f"💾 Results saved to: {args.save_results}")
        
        print(f"\n✅ Principal analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())