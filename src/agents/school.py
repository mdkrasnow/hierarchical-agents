"""
School agent for aggregating and analyzing teacher summaries.

This module implements the third layer of the hierarchical agent system,
consuming TeacherSummary outputs from TeacherAgent and producing 
SchoolSummary outputs with domain statistics, PD cohorts, and narrative summaries.
"""

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agents.base import BaseAgent, AgentResult
from models import TeacherSummary, SchoolSummary, PDCohort, DomainSummary, DomainStatus, RiskLevel, TrendDirection
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


class SchoolInput(BaseModel):
    """Input data for school analysis."""
    teacher_summaries: List[TeacherSummary]
    school_id: Optional[UUID] = None
    school_name: str
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None
    organization_config: Dict[str, Any] = Field(default_factory=dict)
    max_cohort_size: int = 8
    min_cohort_size: int = 3


class DomainSchoolMetrics(BaseModel):
    """Computed metrics for a single domain across all teachers."""
    domain_id: str
    status_distribution: Dict[DomainStatus, int] = Field(default_factory=lambda: {
        DomainStatus.GREEN: 0,
        DomainStatus.YELLOW: 0,
        DomainStatus.RED: 0
    })
    status_percentages: Dict[DomainStatus, float] = Field(default_factory=lambda: {
        DomainStatus.GREEN: 0.0,
        DomainStatus.YELLOW: 0.0,
        DomainStatus.RED: 0.0
    })
    average_score: Optional[float] = None
    teachers_needing_support: List[str] = []
    exemplar_teachers: List[str] = []
    total_teachers: int = 0


class SchoolRiskAnalysis(BaseModel):
    """School-level risk analysis aggregated across teachers."""
    high_risk_teachers: List[str] = []
    medium_risk_teachers: List[str] = []
    low_risk_teachers: List[str] = []
    teachers_needing_immediate_support: List[str] = []
    overall_school_risk: RiskLevel = RiskLevel.LOW
    risk_trend: TrendDirection = TrendDirection.STABLE
    retention_concerns: int = 0


class SchoolNarratives(BaseModel):
    """LLM response format for school narratives."""
    school_strengths: List[str]
    school_needs: List[str] 
    stories_for_principal: List[str]
    stories_for_supervisor: List[str]
    overall_performance_summary: str


class SchoolAgent(BaseAgent):
    """
    Third-layer agent for aggregating teacher summaries into school-level analysis.
    
    Responsibilities:
    - Aggregate TeacherSummary objects for one school
    - Compute domain statistics and distributions
    - Identify PD cohorts based on common needs
    - Generate school-level narratives and performance summaries
    - Identify exemplar teachers and those needing support
    - Produce structured SchoolSummary outputs
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        **kwargs
    ):
        super().__init__(llm_client=llm_client, **kwargs)
        
        # PD topic mapping for cohort formation
        self.pd_topics = {
            'I-A': 'Content Knowledge and Pedagogy',
            'I-B': 'Student Development and Learning',
            'I-C': 'Learning Differences and Accommodations',
            'I-D': 'Instructional Strategies',
            'I-E': 'Assessment Methods',
            'I-F': 'Technology Integration',
            'II-A': 'Creating a Culture of Learning',
            'II-B': 'Classroom Management',
            'II-C': 'Classroom Procedures',
            'II-D': 'Student Behavior Management',
            'II-E': 'Physical Environment',
            'III-A': 'Communication Clarity',
            'III-B': 'Questioning and Discussion',
            'III-C': 'Student Engagement',
            'III-D': 'Assessment in Instruction',
            'III-E': 'Instructional Flexibility',
            'IV-A': 'Professional Reflection',
            'IV-B': 'Data Analysis and Record Keeping',
            'IV-C': 'Communication with Families',
            'IV-D': 'Professional Growth',
            'IV-E': 'School Community Participation',
            'IV-F': 'Professional Responsibilities'
        }
    
    @property
    def agent_type(self) -> str:
        return "SchoolAgent"
    
    @property
    def role_description(self) -> str:
        return "Aggregate teacher summaries into school-level analysis with domain statistics, PD cohorts, and administrative narratives"
    
    async def execute(self, school_input: SchoolInput) -> AgentResult:
        """
        Main execution method for processing school teacher summaries.
        
        Args:
            school_input: Input containing teacher summaries and configuration
            
        Returns:
            AgentResult with SchoolSummary or error information
        """
        try:
            if not school_input.teacher_summaries:
                return AgentResult(
                    success=False,
                    error="No teacher summaries provided for analysis",
                    agent_id=self.agent_id,
                    execution_time_ms=0.0
                )
            
            self.logger.info(f"Processing {len(school_input.teacher_summaries)} teachers for {school_input.school_name}")
            
            # Phase 1: Compute domain statistics across all teachers
            domain_metrics = self._compute_domain_statistics(school_input.teacher_summaries)
            
            # Phase 2: Analyze school-level risk factors
            risk_analysis = self._analyze_school_risk(school_input.teacher_summaries)
            
            # Phase 3: Identify exemplar teachers and those needing support
            teacher_classifications = self._classify_teachers(school_input.teacher_summaries)
            
            # Phase 4: Generate PD cohorts based on common needs
            pd_cohorts = self._generate_pd_cohorts(
                school_input.teacher_summaries,
                school_input.max_cohort_size,
                school_input.min_cohort_size
            )
            
            # Phase 5: Determine priority domains for the school
            priority_domains = self._identify_priority_domains(domain_metrics)
            
            # Phase 6: LLM synthesis for narratives
            narratives = await self._synthesize_school_narratives(
                school_input,
                domain_metrics,
                risk_analysis,
                teacher_classifications,
                priority_domains
            )
            
            # Phase 7: Build comprehensive school summary
            school_summary = self._build_school_summary(
                school_input,
                domain_metrics,
                risk_analysis,
                teacher_classifications,
                pd_cohorts,
                priority_domains,
                narratives
            )
            
            self.logger.info(
                f"Successfully processed school {school_input.school_name}",
                extra={
                    "num_teachers": len(school_input.teacher_summaries),
                    "domains_analyzed": len(domain_metrics),
                    "school_risk_level": risk_analysis.overall_school_risk.value,
                    "pd_cohorts_created": len(pd_cohorts),
                    "exemplar_teachers": len(teacher_classifications.get('exemplar_teachers', []))
                }
            )
            
            return AgentResult(
                success=True,
                data={"school_summary": school_summary.model_dump()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,  # Will be set by tracking
                metadata={
                    "school_name": school_input.school_name,
                    "teachers_count": len(school_input.teacher_summaries),
                    "domains_analyzed": len(domain_metrics)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process school analysis: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={"error_type": type(e).__name__}
            )
    
    def _compute_domain_statistics(self, teacher_summaries: List[TeacherSummary]) -> Dict[str, DomainSchoolMetrics]:
        """
        Compute domain-level statistics across all teachers in the school.
        
        This method performs pure mathematical aggregation without LLM calls.
        """
        domain_metrics = {}
        
        # Collect all domain data across teachers
        domain_data = defaultdict(lambda: {
            'status_counts': Counter(),
            'scores': [],
            'teachers_by_status': defaultdict(list),
            'teacher_names': []
        })
        
        # Aggregate data across all teachers
        for teacher_summary in teacher_summaries:
            for domain_id, domain_summary in teacher_summary.per_domain_overview.items():
                data = domain_data[domain_id]
                
                # Count status distribution
                data['status_counts'][domain_summary.status_color] += 1
                data['teachers_by_status'][domain_summary.status_color].append(teacher_summary.teacher_name)
                
                # Collect scores if available
                if domain_summary.score is not None:
                    data['scores'].append(domain_summary.score)
                
                data['teacher_names'].append(teacher_summary.teacher_name)
        
        # Compute metrics for each domain
        for domain_id, data in domain_data.items():
            metrics = DomainSchoolMetrics(domain_id=domain_id)
            
            total_teachers = sum(data['status_counts'].values())
            metrics.total_teachers = total_teachers
            
            # Status distribution and percentages
            for status in DomainStatus:
                count = data['status_counts'].get(status, 0)
                metrics.status_distribution[status] = count
                metrics.status_percentages[status] = count / total_teachers if total_teachers > 0 else 0.0
            
            # Average score calculation
            if data['scores']:
                metrics.average_score = statistics.mean(data['scores'])
            
            # Identify teachers needing support (Red or Yellow status)
            metrics.teachers_needing_support = (
                data['teachers_by_status'][DomainStatus.RED] +
                data['teachers_by_status'][DomainStatus.YELLOW]
            )
            
            # Identify exemplar teachers (Green status)
            metrics.exemplar_teachers = data['teachers_by_status'][DomainStatus.GREEN]
            
            domain_metrics[domain_id] = metrics
        
        return domain_metrics
    
    def _analyze_school_risk(self, teacher_summaries: List[TeacherSummary]) -> SchoolRiskAnalysis:
        """
        Analyze risk factors across all teachers at the school level.
        
        Performs deterministic aggregation of teacher-level risk signals.
        """
        risk_analysis = SchoolRiskAnalysis()
        
        # Categorize teachers by risk level
        for teacher_summary in teacher_summaries:
            teacher_name = teacher_summary.teacher_name
            
            if teacher_summary.risk_level == RiskLevel.HIGH:
                risk_analysis.high_risk_teachers.append(teacher_name)
            elif teacher_summary.risk_level == RiskLevel.MEDIUM:
                risk_analysis.medium_risk_teachers.append(teacher_name)
            else:
                risk_analysis.low_risk_teachers.append(teacher_name)
            
            if teacher_summary.needs_immediate_support:
                risk_analysis.teachers_needing_immediate_support.append(teacher_name)
        
        # Calculate overall school risk level
        total_teachers = len(teacher_summaries)
        if total_teachers > 0:
            high_risk_rate = len(risk_analysis.high_risk_teachers) / total_teachers
            medium_risk_rate = len(risk_analysis.medium_risk_teachers) / total_teachers
            immediate_support_rate = len(risk_analysis.teachers_needing_immediate_support) / total_teachers
            
            if high_risk_rate >= 0.3 or immediate_support_rate >= 0.2:
                risk_analysis.overall_school_risk = RiskLevel.HIGH
            elif high_risk_rate >= 0.15 or medium_risk_rate >= 0.4:
                risk_analysis.overall_school_risk = RiskLevel.MEDIUM
            else:
                risk_analysis.overall_school_risk = RiskLevel.LOW
        
        # Set retention concerns count
        risk_analysis.retention_concerns = len(risk_analysis.high_risk_teachers) + len(risk_analysis.teachers_needing_immediate_support)
        
        return risk_analysis
    
    def _classify_teachers(self, teacher_summaries: List[TeacherSummary]) -> Dict[str, List[str]]:
        """Classify teachers into performance and support categories."""
        
        exemplar_teachers = []
        teachers_needing_support = []
        high_performers = []
        developing_teachers = []
        
        for teacher_summary in teacher_summaries:
            teacher_name = teacher_summary.teacher_name
            
            # Identify exemplar teachers
            if teacher_summary.is_exemplar:
                exemplar_teachers.append(teacher_name)
                high_performers.append(teacher_name)
            elif teacher_summary.needs_immediate_support or teacher_summary.risk_level == RiskLevel.HIGH:
                teachers_needing_support.append(teacher_name)
            else:
                # Check domain distribution for performance level
                green_count = teacher_summary.domain_distribution.get(DomainStatus.GREEN, 0)
                red_count = teacher_summary.domain_distribution.get(DomainStatus.RED, 0)
                total_domains = sum(teacher_summary.domain_distribution.values())
                
                if total_domains > 0:
                    green_rate = green_count / total_domains
                    red_rate = red_count / total_domains
                    
                    if green_rate >= 0.6 and red_rate <= 0.1:
                        high_performers.append(teacher_name)
                    elif red_rate >= 0.3 or teacher_summary.risk_level == RiskLevel.MEDIUM:
                        developing_teachers.append(teacher_name)
                    else:
                        developing_teachers.append(teacher_name)
        
        return {
            "exemplar_teachers": exemplar_teachers,
            "teachers_needing_support": teachers_needing_support,
            "high_performers": high_performers,
            "developing_teachers": developing_teachers
        }
    
    def _generate_pd_cohorts(
        self,
        teacher_summaries: List[TeacherSummary],
        max_cohort_size: int = 8,
        min_cohort_size: int = 3
    ) -> List[PDCohort]:
        """
        Generate Professional Development cohorts based on common teacher needs.
        
        Groups teachers by shared PD domain needs while avoiding oversized cohorts.
        """
        # Count teachers needing each domain
        domain_needs = defaultdict(list)
        
        for teacher_summary in teacher_summaries:
            teacher_name = teacher_summary.teacher_name
            teacher_id = teacher_summary.teacher_id
            
            # Add teachers who have domains needing PD focus
            for domain_id in teacher_summary.recommended_PD_domains:
                domain_needs[domain_id].append({
                    'teacher_id': teacher_id,
                    'teacher_name': teacher_name
                })
        
        # Create cohorts for domains with sufficient teachers
        pd_cohorts = []
        
        # Sort domains by number of teachers needing support (prioritize high-need areas)
        sorted_domains = sorted(domain_needs.items(), key=lambda x: len(x[1]), reverse=True)
        
        for domain_id, teacher_list in sorted_domains:
            if len(teacher_list) >= min_cohort_size:
                # If too many teachers, create multiple cohorts
                if len(teacher_list) > max_cohort_size:
                    # Split into multiple balanced cohorts
                    num_cohorts = (len(teacher_list) + max_cohort_size - 1) // max_cohort_size
                    cohort_size = len(teacher_list) // num_cohorts
                    
                    for i in range(num_cohorts):
                        start_idx = i * cohort_size
                        end_idx = start_idx + cohort_size if i < num_cohorts - 1 else len(teacher_list)
                        cohort_teachers = teacher_list[start_idx:end_idx]
                        
                        cohort = PDCohort(
                            domain_id=domain_id,
                            focus_area=self.pd_topics.get(domain_id, f"Domain {domain_id} Development"),
                            teacher_ids=[t['teacher_id'] for t in cohort_teachers if t['teacher_id']],
                            teacher_names=[t['teacher_name'] for t in cohort_teachers],
                            priority_level=self._determine_cohort_priority(domain_id, len(teacher_list)),
                            suggested_duration=self._suggest_pd_duration(domain_id, len(cohort_teachers))
                        )
                        pd_cohorts.append(cohort)
                else:
                    # Single cohort for this domain
                    cohort = PDCohort(
                        domain_id=domain_id,
                        focus_area=self.pd_topics.get(domain_id, f"Domain {domain_id} Development"),
                        teacher_ids=[t['teacher_id'] for t in teacher_list if t['teacher_id']],
                        teacher_names=[t['teacher_name'] for t in teacher_list],
                        priority_level=self._determine_cohort_priority(domain_id, len(teacher_list)),
                        suggested_duration=self._suggest_pd_duration(domain_id, len(teacher_list))
                    )
                    pd_cohorts.append(cohort)
        
        # Sort cohorts by priority and then by size
        priority_order = {"high": 0, "medium": 1, "low": 2}
        pd_cohorts.sort(key=lambda c: (priority_order.get(c.priority_level, 2), -len(c.teacher_names)))
        
        return pd_cohorts
    
    def _determine_cohort_priority(self, domain_id: str, teacher_count: int) -> str:
        """Determine priority level for a PD cohort based on domain and teacher count."""
        total_teachers = teacher_count  # This would ideally be total school teachers, but we use cohort size as proxy
        
        # High priority if many teachers need this domain or if it's a critical teaching domain
        critical_domains = ['III-A', 'III-B', 'III-C', 'II-B', 'I-A']  # Core instructional and management
        
        if domain_id in critical_domains or teacher_count >= 6:
            return "high"
        elif teacher_count >= 4:
            return "medium"
        else:
            return "low"
    
    def _suggest_pd_duration(self, domain_id: str, cohort_size: int) -> str:
        """Suggest PD duration based on domain complexity and cohort size."""
        complex_domains = ['I-A', 'I-D', 'III-B', 'III-C', 'IV-A']  # Domains requiring deeper work
        
        if domain_id in complex_domains:
            return "6-8 sessions" if cohort_size >= 5 else "4-6 sessions"
        else:
            return "3-4 sessions" if cohort_size >= 5 else "2-3 sessions"
    
    def _identify_priority_domains(self, domain_metrics: Dict[str, DomainSchoolMetrics]) -> List[str]:
        """Identify school-wide priority domains based on need and impact."""
        
        domain_priorities = []
        
        for domain_id, metrics in domain_metrics.items():
            priority_score = 0
            
            if metrics.total_teachers == 0:
                continue
            
            # Factor 1: High percentage of teachers needing support
            need_rate = (metrics.status_distribution[DomainStatus.RED] + 
                        metrics.status_distribution[DomainStatus.YELLOW]) / metrics.total_teachers
            
            if need_rate >= 0.5:
                priority_score += 3
            elif need_rate >= 0.3:
                priority_score += 2
            elif need_rate >= 0.2:
                priority_score += 1
            
            # Factor 2: Low average performance
            if metrics.average_score is not None:
                if metrics.average_score < 2.0:
                    priority_score += 3
                elif metrics.average_score < 2.5:
                    priority_score += 2
                elif metrics.average_score < 3.0:
                    priority_score += 1
            
            # Factor 3: Critical domains get bonus
            critical_domains = ['III-A', 'III-B', 'III-C', 'II-B', 'I-A']
            if domain_id in critical_domains:
                priority_score += 1
            
            if priority_score > 0:
                domain_priorities.append((domain_id, priority_score))
        
        # Sort by priority score and return top domains
        domain_priorities.sort(key=lambda x: x[1], reverse=True)
        return [domain_id for domain_id, score in domain_priorities[:5]]
    
    async def _synthesize_school_narratives(
        self,
        school_input: SchoolInput,
        domain_metrics: Dict[str, DomainSchoolMetrics],
        risk_analysis: SchoolRiskAnalysis,
        teacher_classifications: Dict[str, List[str]],
        priority_domains: List[str]
    ) -> Dict[str, Any]:
        """
        Use LLM to synthesize school-level narratives.
        
        Creates strength/need narratives and principal/supervisor stories.
        """
        try:
            # Prepare context for LLM
            context = self._prepare_school_narrative_context(
                school_input, domain_metrics, risk_analysis, teacher_classifications, priority_domains
            )
            
            # Generate narratives using LLM
            narratives = await self.llm_call_with_template(
                template_name="school_narratives",
                template_variables={
                    "school_name": school_input.school_name,
                    "teacher_count": len(school_input.teacher_summaries),
                    "domain_analysis": context["domain_analysis"],
                    "risk_analysis": context["risk_summary"],
                    "teacher_performance": context["teacher_performance"],
                    "priority_areas": context["priority_areas"]
                },
                response_format=SchoolNarratives
            )
            
            if isinstance(narratives, SchoolNarratives):
                return {
                    "school_strengths": narratives.school_strengths,
                    "school_needs": narratives.school_needs,
                    "stories_for_principal": narratives.stories_for_principal,
                    "stories_for_supervisor": narratives.stories_for_supervisor,
                    "overall_performance_summary": narratives.overall_performance_summary
                }
            
        except Exception as e:
            self.logger.warning(f"LLM narrative synthesis failed: {e}")
        
        # Fallback: Generate narratives programmatically
        return self._generate_fallback_school_narratives(
            domain_metrics, risk_analysis, teacher_classifications, priority_domains
        )
    
    def _prepare_school_narrative_context(
        self,
        school_input: SchoolInput,
        domain_metrics: Dict[str, DomainSchoolMetrics],
        risk_analysis: SchoolRiskAnalysis,
        teacher_classifications: Dict[str, List[str]],
        priority_domains: List[str]
    ) -> Dict[str, str]:
        """Prepare structured context for school narrative generation."""
        
        # Domain analysis summary
        strong_domains = [d for d, m in domain_metrics.items() if m.status_percentages[DomainStatus.GREEN] >= 0.6]
        weak_domains = priority_domains[:3]  # Top 3 priority domains
        
        domain_analysis = f"Strong performance across: {', '.join(strong_domains[:3]) if strong_domains else 'No consistently strong school-wide domains'}. "
        domain_analysis += f"Areas needing school-wide focus: {', '.join(weak_domains) if weak_domains else 'Generally balanced performance'}."
        
        # Risk summary
        total_teachers = len(school_input.teacher_summaries)
        risk_summary = f"School risk level: {risk_analysis.overall_school_risk.value}. "
        if risk_analysis.high_risk_teachers:
            risk_summary += f"{len(risk_analysis.high_risk_teachers)}/{total_teachers} teachers at high risk. "
        if risk_analysis.teachers_needing_immediate_support:
            risk_summary += f"{len(risk_analysis.teachers_needing_immediate_support)} need immediate support. "
        
        # Teacher performance summary
        exemplar_count = len(teacher_classifications.get('exemplar_teachers', []))
        support_count = len(teacher_classifications.get('teachers_needing_support', []))
        teacher_performance = f"Teacher performance: {exemplar_count} exemplar, {support_count} needing support out of {total_teachers} total."
        
        # Priority areas
        priority_areas = f"School-wide priority domains: {', '.join(priority_domains[:3])}." if priority_domains else "No major priority areas identified."
        
        return {
            "domain_analysis": domain_analysis,
            "risk_summary": risk_summary,
            "teacher_performance": teacher_performance,
            "priority_areas": priority_areas
        }
    
    def _generate_fallback_school_narratives(
        self,
        domain_metrics: Dict[str, DomainSchoolMetrics],
        risk_analysis: SchoolRiskAnalysis,
        teacher_classifications: Dict[str, List[str]],
        priority_domains: List[str]
    ) -> Dict[str, Any]:
        """Generate basic school narratives when LLM synthesis fails."""
        
        # School strengths
        strong_domains = [d for d, m in domain_metrics.items() if m.status_percentages[DomainStatus.GREEN] >= 0.6]
        school_strengths = []
        if strong_domains:
            school_strengths.append(f"Consistent strength in {', '.join(strong_domains[:2])}")
        if len(teacher_classifications.get('exemplar_teachers', [])) > 0:
            school_strengths.append(f"{len(teacher_classifications['exemplar_teachers'])} exemplar teachers demonstrating best practices")
        if risk_analysis.overall_school_risk == RiskLevel.LOW:
            school_strengths.append("Strong teacher retention and wellness indicators")
        
        # School needs  
        school_needs = []
        if priority_domains:
            school_needs.append(f"Professional development focus needed in {', '.join(priority_domains[:2])}")
        if len(teacher_classifications.get('teachers_needing_support', [])) > 0:
            school_needs.append(f"Targeted support for {len(teacher_classifications['teachers_needing_support'])} teachers")
        if risk_analysis.overall_school_risk != RiskLevel.LOW:
            school_needs.append("Teacher wellness and retention support strategies")
        
        # Stories for principal
        stories_for_principal = []
        if teacher_classifications.get('exemplar_teachers'):
            stories_for_principal.append(f"Leverage {', '.join(teacher_classifications['exemplar_teachers'][:2])} as mentor teachers and instructional leaders")
        if priority_domains:
            stories_for_principal.append(f"Focus PD resources on {priority_domains[0]} to achieve school-wide improvement")
        
        # Stories for supervisor
        stories_for_supervisor = []
        school_summary = "performing adequately"
        if risk_analysis.overall_school_risk == RiskLevel.LOW and strong_domains:
            school_summary = "demonstrating strong instructional practices"
        elif risk_analysis.overall_school_risk == RiskLevel.HIGH or len(priority_domains) >= 3:
            school_summary = "requiring focused support and intervention"
        
        stories_for_supervisor.append(f"School is {school_summary} with clear action plan for continued growth")
        
        # Overall performance summary
        if strong_domains and risk_analysis.overall_school_risk == RiskLevel.LOW:
            performance_summary = "School demonstrates strong overall performance with effective teaching practices and stable staff."
        elif priority_domains and risk_analysis.overall_school_risk != RiskLevel.LOW:
            performance_summary = "School requires targeted support in key instructional areas and teacher wellness initiatives."
        else:
            performance_summary = "School shows mixed performance with opportunities for growth in specific domains."
        
        return {
            "school_strengths": school_strengths or ["Committed teaching staff working toward improvement"],
            "school_needs": school_needs or ["Continued focus on instructional excellence"],
            "stories_for_principal": stories_for_principal or ["Continue supporting teacher growth and collaboration"],
            "stories_for_supervisor": stories_for_supervisor,
            "overall_performance_summary": performance_summary
        }
    
    def _build_school_summary(
        self,
        school_input: SchoolInput,
        domain_metrics: Dict[str, DomainSchoolMetrics],
        risk_analysis: SchoolRiskAnalysis,
        teacher_classifications: Dict[str, List[str]],
        pd_cohorts: List[PDCohort],
        priority_domains: List[str],
        narratives: Dict[str, Any]
    ) -> SchoolSummary:
        """Build the final SchoolSummary output."""
        
        # Build domain statistics and percentages
        domain_stats = {}
        domain_percentages = {}
        
        for domain_id, metrics in domain_metrics.items():
            domain_stats[domain_id] = metrics.status_distribution.copy()
            domain_percentages[domain_id] = metrics.status_percentages.copy()
        
        # Determine overall school performance level
        total_teachers = len(school_input.teacher_summaries)
        if total_teachers > 0:
            # Calculate weighted average of green percentages across domains
            green_rates = [m.status_percentages[DomainStatus.GREEN] for m in domain_metrics.values()]
            red_rates = [m.status_percentages[DomainStatus.RED] for m in domain_metrics.values()]
            
            avg_green_rate = statistics.mean(green_rates) if green_rates else 0
            avg_red_rate = statistics.mean(red_rates) if red_rates else 0
            
            if avg_green_rate >= 0.6 and avg_red_rate <= 0.15:
                overall_performance = DomainStatus.GREEN
            elif avg_red_rate >= 0.3 or avg_green_rate <= 0.3:
                overall_performance = DomainStatus.RED
            else:
                overall_performance = DomainStatus.YELLOW
        else:
            overall_performance = DomainStatus.YELLOW
        
        # Determine improvement trend (simplified - would need historical data for real trend)
        improvement_trend = TrendDirection.STABLE
        if risk_analysis.overall_school_risk == RiskLevel.LOW and len(teacher_classifications.get('exemplar_teachers', [])) > total_teachers * 0.2:
            improvement_trend = TrendDirection.IMPROVING
        elif risk_analysis.overall_school_risk == RiskLevel.HIGH or len(priority_domains) >= 4:
            improvement_trend = TrendDirection.DECLINING
        
        return SchoolSummary(
            school_id=school_input.school_id,
            school_name=school_input.school_name,
            organization_id=None,  # Would need to be passed in school_input if available
            analysis_period_start=school_input.analysis_period_start,
            analysis_period_end=school_input.analysis_period_end,
            num_teachers_analyzed=len(school_input.teacher_summaries),
            domain_stats=domain_stats,
            domain_percentages=domain_percentages,
            PD_cohorts=pd_cohorts,
            priority_domains=priority_domains,
            school_strengths=narratives.get("school_strengths", []),
            school_needs=narratives.get("school_needs", []),
            stories_for_principal=narratives.get("stories_for_principal", []),
            stories_for_supervisor_or_board=narratives.get("stories_for_supervisor", []),
            exemplar_teachers=teacher_classifications.get("exemplar_teachers", []),
            teachers_needing_support=teacher_classifications.get("teachers_needing_support", []),
            overall_performance_level=overall_performance,
            school_risk_level=risk_analysis.overall_school_risk,
            improvement_trend=improvement_trend
        )