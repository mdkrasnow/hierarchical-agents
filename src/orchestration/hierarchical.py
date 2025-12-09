"""
Hierarchical orchestrator for coordinating multi-layer agent execution.

This module implements the main orchestration logic for running the hierarchical
agent chain: Evaluation → Teacher → School → District agents based on user role
and permissions.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field, validator

from agents import DanielsonEvaluationAgent, TeacherAgent, SchoolAgent, DistrictAgent
from agents.base import AgentConfig, AgentResult
from agents.evaluation import EvaluationInput
from agents.teacher import TeacherInput
from agents.school import SchoolInput
from agents.district import DistrictInput
from models import (
    User, Organization, DanielsonEvaluation, 
    EvaluationSummary, TeacherSummary, SchoolSummary, DistrictSummary
)
from models.permissions import UserScope, PermissionScope, UserRole
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


class OrchestrationConfig(BaseModel):
    """Configuration for hierarchical orchestration."""
    
    # Concurrency limits
    max_concurrent_evaluations: int = 10
    max_concurrent_teachers: int = 5
    max_concurrent_schools: int = 3
    
    # Analysis settings
    max_history_months: Optional[int] = 24
    include_informal_evaluations: bool = True
    include_archived_evaluations: bool = False
    
    # Output preferences
    verbose_output: bool = False
    include_metrics: bool = True
    save_intermediate_results: bool = False
    
    # Agent configurations
    evaluation_agent_config: AgentConfig = Field(default_factory=AgentConfig)
    teacher_agent_config: AgentConfig = Field(default_factory=AgentConfig)
    school_agent_config: AgentConfig = Field(default_factory=AgentConfig)
    district_agent_config: AgentConfig = Field(default_factory=AgentConfig)
    
    # Terminal output settings
    show_progress: bool = True
    summary_first: bool = True
    
    @validator('max_concurrent_evaluations', 'max_concurrent_teachers', 'max_concurrent_schools')
    def validate_concurrency_limits(cls, v):
        if v < 1 or v > 50:
            raise ValueError("Concurrency limits must be between 1 and 50")
        return v


class OrchestrationResult(BaseModel):
    """Results from hierarchical orchestration."""
    
    # Execution metadata
    success: bool
    user_id: UUID
    user_role: UserRole
    execution_time_ms: float
    start_time: datetime
    end_time: datetime
    
    # Data processed
    num_evaluations_processed: int = 0
    num_teachers_analyzed: int = 0
    num_schools_analyzed: int = 0
    
    # Results by level (only populated based on user role)
    evaluation_summaries: List[EvaluationSummary] = []
    teacher_summaries: List[TeacherSummary] = []
    school_summaries: List[SchoolSummary] = []
    district_summary: Optional[DistrictSummary] = None
    
    # Primary result for user
    primary_result: Optional[Dict[str, Any]] = None
    primary_result_type: str = "none"  # evaluation, teacher, school, district
    
    # Execution details
    agent_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = []
    warnings: List[str] = []
    
    # Terminal summary
    executive_summary: str = ""
    key_insights: List[str] = []
    recommended_actions: List[str] = []


class HierarchicalOrchestrator:
    """
    Orchestrator for running hierarchical agent chains based on user role and scope.
    
    Coordinates execution of Evaluation → Teacher → School → District agents
    with appropriate filtering and permissions based on user role.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[OrchestrationConfig] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        self.llm_client = llm_client
        self.config = config or OrchestrationConfig()
        self.logger = logger_instance or logger
        
        # Initialize agents
        self._init_agents()
        
        # Execution tracking
        self._start_time: Optional[datetime] = None
        self._execution_metrics = defaultdict(list)
    
    def _init_agents(self):
        """Initialize all agent instances with configurations."""
        self.evaluation_agent = DanielsonEvaluationAgent(
            llm_client=self.llm_client,
            config=self.config.evaluation_agent_config
        )
        
        self.teacher_agent = TeacherAgent(
            llm_client=self.llm_client,
            config=self.config.teacher_agent_config
        )
        
        self.school_agent = SchoolAgent(
            llm_client=self.llm_client,
            config=self.config.school_agent_config
        )
        
        self.district_agent = DistrictAgent(
            llm_client=self.llm_client,
            config=self.config.district_agent_config
        )
    
    async def execute_for_user(
        self,
        user_id: UUID,
        user_scope: UserScope,
        organization: Organization,
        evaluations: List[DanielsonEvaluation]
    ) -> OrchestrationResult:
        """
        Execute hierarchical analysis for a specific user based on their role and scope.
        
        Args:
            user_id: ID of the user requesting analysis
            user_scope: User's permissions and scope
            organization: Organization context
            evaluations: List of evaluations to process
            
        Returns:
            OrchestrationResult with appropriate level of analysis
        """
        self._start_time = datetime.now()
        execution_start = time.time()
        
        if self.config.show_progress:
            print(f"\n🚀 Starting hierarchical analysis for {user_scope.role.value}")
            print(f"📊 Processing {len(evaluations)} evaluations...")
        
        try:
            # Determine execution path based on user role
            hierarchy_config = user_scope.get_agent_hierarchy_config()
            
            # Phase 1: Process evaluations (always required)
            if self.config.show_progress:
                print(f"🔍 Phase 1: Processing {len(evaluations)} individual evaluations...")
            
            evaluation_summaries = await self._process_evaluations(
                evaluations, 
                organization,
                user_scope
            )
            
            result = OrchestrationResult(
                success=True,
                user_id=user_id,
                user_role=user_scope.role,
                execution_time_ms=0.0,  # Will be calculated at end
                start_time=self._start_time,
                end_time=datetime.now(),
                num_evaluations_processed=len(evaluation_summaries),
                evaluation_summaries=evaluation_summaries
            )
            
            # Phase 2: Teacher analysis (if configured)
            teacher_summaries = []
            if hierarchy_config.get("run_teacher_agent"):
                if self.config.show_progress:
                    print(f"👨‍🏫 Phase 2: Analyzing teachers...")
                
                teacher_summaries = await self._process_teachers(
                    evaluation_summaries,
                    organization,
                    user_scope
                )
                result.teacher_summaries = teacher_summaries
                result.num_teachers_analyzed = len(teacher_summaries)
            
            # Phase 3: School analysis (if configured)
            school_summaries = []
            if hierarchy_config.get("run_school_agent") and teacher_summaries:
                if self.config.show_progress:
                    print(f"🏫 Phase 3: Analyzing schools...")
                
                school_summaries = await self._process_schools(
                    teacher_summaries,
                    organization,
                    user_scope
                )
                result.school_summaries = school_summaries
                result.num_schools_analyzed = len(school_summaries)
            
            # Phase 4: District analysis (if configured)
            district_summary = None
            if hierarchy_config.get("run_district_agent") and school_summaries:
                if self.config.show_progress:
                    print(f"🌆 Phase 4: Analyzing district...")
                
                district_summary = await self._process_district(
                    school_summaries,
                    organization,
                    user_scope
                )
                result.district_summary = district_summary
            
            # Set primary result based on user role
            result.primary_result, result.primary_result_type = self._determine_primary_result(
                hierarchy_config,
                evaluation_summaries,
                teacher_summaries,
                school_summaries,
                district_summary
            )
            
            # Generate executive summary and insights
            result.executive_summary, result.key_insights, result.recommended_actions = (
                await self._generate_executive_summary(result, user_scope, organization)
            )
            
            # Calculate final metrics
            result.execution_time_ms = (time.time() - execution_start) * 1000
            result.end_time = datetime.now()
            
            if self.config.include_metrics:
                result.agent_metrics = self._collect_agent_metrics()
            
            if self.config.show_progress:
                print(f"✅ Analysis complete! ({result.execution_time_ms/1000:.1f}s)")
                if self.config.summary_first:
                    self._print_terminal_summary(result)
            
            self.logger.info(
                f"Hierarchical orchestration completed for user {user_id}",
                extra={
                    "user_role": user_scope.role.value,
                    "execution_time_ms": result.execution_time_ms,
                    "evaluations_processed": result.num_evaluations_processed,
                    "teachers_analyzed": result.num_teachers_analyzed,
                    "schools_analyzed": result.num_schools_analyzed,
                    "primary_result_type": result.primary_result_type
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            
            error_result = OrchestrationResult(
                success=False,
                user_id=user_id,
                user_role=user_scope.role,
                execution_time_ms=execution_time,
                start_time=self._start_time,
                end_time=datetime.now(),
                errors=[str(e)],
                executive_summary=f"Analysis failed: {str(e)}"
            )
            
            self.logger.error(
                f"Hierarchical orchestration failed for user {user_id}: {e}",
                exc_info=True,
                extra={"user_role": user_scope.role.value}
            )
            
            return error_result
    
    async def _process_evaluations(
        self,
        evaluations: List[DanielsonEvaluation],
        organization: Organization,
        user_scope: UserScope
    ) -> List[EvaluationSummary]:
        """Process individual evaluations with concurrency control."""
        
        # Create evaluation inputs
        evaluation_inputs = []
        for evaluation in evaluations:
            eval_input = EvaluationInput(
                evaluation_data=evaluation,
                organization_config=organization.performance_level_config or {},
                analysis_focus="comprehensive",
                include_evidence=True,
                max_evidence_snippets=5
            )
            evaluation_inputs.append(eval_input)
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)
        
        async def process_single_evaluation(eval_input: EvaluationInput) -> Optional[EvaluationSummary]:
            async with semaphore:
                try:
                    result = await self.evaluation_agent.execute_with_tracking(
                        evaluation_input=eval_input
                    )
                    if result.success and result.data:
                        # Convert result data to EvaluationSummary
                        eval_summary_data = result.data.get("evaluation_summary")
                        if eval_summary_data:
                            return EvaluationSummary(**eval_summary_data)
                    return None
                except Exception as e:
                    self.logger.warning(f"Failed to process evaluation {eval_input.evaluation_data.id}: {e}")
                    return None
        
        # Execute all evaluation tasks
        tasks = [process_single_evaluation(eval_input) for eval_input in evaluation_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        evaluation_summaries = []
        for result in results:
            if isinstance(result, EvaluationSummary):
                evaluation_summaries.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Evaluation processing task failed: {result}")
        
        return evaluation_summaries
    
    async def _process_teachers(
        self,
        evaluation_summaries: List[EvaluationSummary],
        organization: Organization,
        user_scope: UserScope
    ) -> List[TeacherSummary]:
        """Process teacher summaries with concurrency control."""
        
        # Group evaluations by teacher
        teacher_evaluations = defaultdict(list)
        for eval_summary in evaluation_summaries:
            teacher_key = (eval_summary.teacher_name, eval_summary.school_name)
            teacher_evaluations[teacher_key].append(eval_summary)
        
        # Create teacher inputs
        teacher_inputs = []
        for (teacher_name, school_name), evaluations in teacher_evaluations.items():
            teacher_input = TeacherInput(
                evaluations=evaluations,
                teacher_id=evaluations[0].teacher_id,
                teacher_name=teacher_name,
                organization_config=organization.performance_level_config or {},
                pd_focus_limit=3
            )
            teacher_inputs.append(teacher_input)
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_teachers)
        
        async def process_single_teacher(teacher_input: TeacherInput) -> Optional[TeacherSummary]:
            async with semaphore:
                try:
                    result = await self.teacher_agent.execute_with_tracking(
                        teacher_input=teacher_input
                    )
                    if result.success and result.data:
                        teacher_summary_data = result.data.get("teacher_summary")
                        if teacher_summary_data:
                            return TeacherSummary(**teacher_summary_data)
                    return None
                except Exception as e:
                    self.logger.warning(f"Failed to process teacher {teacher_input.teacher_name}: {e}")
                    return None
        
        # Execute all teacher tasks
        tasks = [process_single_teacher(teacher_input) for teacher_input in teacher_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        teacher_summaries = []
        for result in results:
            if isinstance(result, TeacherSummary):
                teacher_summaries.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Teacher processing task failed: {result}")
        
        return teacher_summaries
    
    async def _process_schools(
        self,
        teacher_summaries: List[TeacherSummary],
        organization: Organization,
        user_scope: UserScope
    ) -> List[SchoolSummary]:
        """Process school summaries with concurrency control."""
        
        # Group teachers by school
        school_teachers = defaultdict(list)
        for teacher_summary in teacher_summaries:
            school_key = (teacher_summary.school_name, teacher_summary.school_id)
            school_teachers[school_key].append(teacher_summary)
        
        # Create school inputs
        school_inputs = []
        for (school_name, school_id), teachers in school_teachers.items():
            school_input = SchoolInput(
                teacher_summaries=teachers,
                school_id=school_id,
                school_name=school_name,
                organization_config=organization.performance_level_config or {},
                max_cohort_size=8,
                min_cohort_size=3
            )
            school_inputs.append(school_input)
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_schools)
        
        async def process_single_school(school_input: SchoolInput) -> Optional[SchoolSummary]:
            async with semaphore:
                try:
                    result = await self.school_agent.execute_with_tracking(
                        school_input=school_input
                    )
                    if result.success and result.data:
                        school_summary_data = result.data.get("school_summary")
                        if school_summary_data:
                            return SchoolSummary(**school_summary_data)
                    return None
                except Exception as e:
                    self.logger.warning(f"Failed to process school {school_input.school_name}: {e}")
                    return None
        
        # Execute all school tasks
        tasks = [process_single_school(school_input) for school_input in school_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        school_summaries = []
        for result in results:
            if isinstance(result, SchoolSummary):
                school_summaries.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"School processing task failed: {result}")
        
        return school_summaries
    
    async def _process_district(
        self,
        school_summaries: List[SchoolSummary],
        organization: Organization,
        user_scope: UserScope
    ) -> Optional[DistrictSummary]:
        """Process district summary."""
        
        district_input = DistrictInput(
            school_summaries=school_summaries,
            organization_id=organization.id,
            organization_name=organization.name,
            organization_config=organization.performance_level_config or {},
            max_board_stories=6
        )
        
        try:
            result = await self.district_agent.execute_with_tracking(
                district_input=district_input
            )
            if result.success and result.data:
                district_summary_data = result.data.get("district_summary")
                if district_summary_data:
                    return DistrictSummary(**district_summary_data)
            return None
        except Exception as e:
            self.logger.warning(f"Failed to process district {organization.name}: {e}")
            return None
    
    def _determine_primary_result(
        self,
        hierarchy_config: Dict[str, Any],
        evaluation_summaries: List[EvaluationSummary],
        teacher_summaries: List[TeacherSummary],
        school_summaries: List[SchoolSummary],
        district_summary: Optional[DistrictSummary]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Determine the primary result to return based on user role."""
        
        primary_output = hierarchy_config.get("primary_output", "evaluation")
        
        if primary_output == "district_summary" and district_summary:
            return district_summary.model_dump(), "district"
        elif primary_output == "school_summary" and school_summaries:
            # For principals, return their school(s)
            if len(school_summaries) == 1:
                return school_summaries[0].model_dump(), "school"
            else:
                return {"schools": [school.model_dump() for school in school_summaries]}, "school"
        elif primary_output == "teacher_summary" and teacher_summaries:
            return {"teachers": [teacher.model_dump() for teacher in teacher_summaries]}, "teacher"
        elif evaluation_summaries:
            return {"evaluations": [eval.model_dump() for eval in evaluation_summaries]}, "evaluation"
        
        return None, "none"
    
    async def _generate_executive_summary(
        self,
        result: OrchestrationResult,
        user_scope: UserScope,
        organization: Organization
    ) -> Tuple[str, List[str], List[str]]:
        """Generate executive summary and key insights for the user."""
        
        # Basic summary based on what was processed
        summary_parts = []
        insights = []
        actions = []
        
        # Role-specific summary
        if user_scope.role == UserRole.SUPERINTENDENT and result.district_summary:
            summary_parts.append(
                f"District analysis completed for {organization.name}: "
                f"{result.num_schools_analyzed} schools, {result.num_teachers_analyzed} teachers processed."
            )
            
            if result.district_summary.overall_district_health == "green":
                insights.append("District demonstrates strong overall performance")
            elif result.district_summary.overall_district_health == "red":
                insights.append("District requires focused intervention and support")
            
            if result.district_summary.priority_domains:
                actions.append(f"Focus system PD on {result.district_summary.priority_domains[0]}")
            
            if result.district_summary.schools_needing_support:
                actions.append(f"Deploy support teams to {len(result.district_summary.schools_needing_support)} schools")
        
        elif user_scope.role == UserRole.PRINCIPAL and result.school_summaries:
            school = result.school_summaries[0] if len(result.school_summaries) == 1 else None
            summary_parts.append(
                f"School analysis completed: {result.num_teachers_analyzed} teachers processed."
            )
            
            if school:
                if school.overall_performance_level == "green":
                    insights.append("School demonstrates strong instructional practices")
                elif school.overall_performance_level == "red":
                    insights.append("School requires targeted instructional support")
                
                if school.priority_domains:
                    actions.append(f"Prioritize PD in {school.priority_domains[0]}")
                
                if school.teachers_needing_support:
                    actions.append(f"Provide intensive support for {len(school.teachers_needing_support)} teachers")
        
        else:
            # Evaluator or teacher-level analysis
            summary_parts.append(
                f"Teacher analysis completed: {result.num_teachers_analyzed} teachers, "
                f"{result.num_evaluations_processed} evaluations processed."
            )
            
            if result.teacher_summaries:
                high_risk_teachers = [t for t in result.teacher_summaries if t.risk_level == "high"]
                exemplar_teachers = [t for t in result.teacher_summaries if t.is_exemplar]
                
                if exemplar_teachers:
                    insights.append(f"{len(exemplar_teachers)} exemplar teachers identified")
                if high_risk_teachers:
                    insights.append(f"{len(high_risk_teachers)} teachers need immediate support")
                    actions.append("Develop targeted support plans for high-risk teachers")
        
        executive_summary = " ".join(summary_parts) if summary_parts else "Analysis completed successfully."
        
        # Add timing insight
        insights.append(f"Analysis completed in {result.execution_time_ms/1000:.1f} seconds")
        
        return executive_summary, insights, actions
    
    def _collect_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect performance metrics from all agents."""
        metrics = {}
        
        for agent_name, agent in [
            ("evaluation", self.evaluation_agent),
            ("teacher", self.teacher_agent),
            ("school", self.school_agent),
            ("district", self.district_agent)
        ]:
            if agent.metrics:
                metrics[agent_name] = agent._get_metrics_dict()
        
        return metrics
    
    def _print_terminal_summary(self, result: OrchestrationResult):
        """Print a terminal-friendly summary for immediate feedback."""
        print("\n" + "="*60)
        print(f"📋 HIERARCHICAL ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"👤 Role: {result.user_role.value.title()}")
        print(f"⏱️  Execution Time: {result.execution_time_ms/1000:.1f}s")
        print(f"📊 Data Processed: {result.num_evaluations_processed} evaluations, "
              f"{result.num_teachers_analyzed} teachers, {result.num_schools_analyzed} schools")
        
        if result.executive_summary:
            print(f"\n📝 Executive Summary:")
            print(f"   {result.executive_summary}")
        
        if result.key_insights:
            print(f"\n💡 Key Insights:")
            for insight in result.key_insights[:3]:  # Show top 3
                print(f"   • {insight}")
        
        if result.recommended_actions:
            print(f"\n🎯 Recommended Actions:")
            for action in result.recommended_actions[:3]:  # Show top 3
                print(f"   • {action}")
        
        print("\n" + "="*60)