"""
District agent for aggregating and analyzing school summaries.

This module implements the top layer of the hierarchical agent system,
consuming SchoolSummary outputs from SchoolAgent and producing 
DistrictSummary outputs with cross-school comparisons, system-level PD strategy, 
and board-ready stories.
"""

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agents.base import BaseAgent, AgentResult
from models import (
    SchoolSummary, 
    DistrictSummary, 
    SchoolRanking, 
    BoardStory, 
    DomainStatus, 
    RiskLevel, 
    TrendDirection
)
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


class DistrictInput(BaseModel):
    """Input data for district analysis."""
    school_summaries: List[SchoolSummary]
    organization_id: UUID
    organization_name: str
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None
    organization_config: Dict[str, Any] = Field(default_factory=dict)
    max_board_stories: int = 6
    min_school_count_for_ranking: int = 3


class DomainDistrictMetrics(BaseModel):
    """Computed metrics for a single domain across all schools."""
    domain_id: str
    school_averages: Dict[str, float] = Field(default_factory=dict)  # school_name -> avg score
    district_average: Optional[float] = None
    performance_distribution: Dict[DomainStatus, int] = Field(default_factory=lambda: {
        DomainStatus.GREEN: 0,
        DomainStatus.YELLOW: 0,
        DomainStatus.RED: 0
    })
    top_performing_schools: List[str] = []
    schools_needing_support: List[str] = []
    total_schools_analyzed: int = 0
    total_teachers_analyzed: int = 0


class SystemPDAnalysis(BaseModel):
    """District-wide PD needs analysis."""
    most_common_needs: Dict[str, int] = Field(default_factory=dict)  # domain -> num teachers needing
    shared_priority_domains: List[str] = []  # domains needed across multiple schools
    school_specific_needs: Dict[str, List[str]] = Field(default_factory=dict)  # school -> unique needs
    recommended_initiatives: List[str] = []
    cohort_opportunities: List[str] = []  # Cross-school cohort possibilities


class DistrictRiskAnalysis(BaseModel):
    """District-level risk and wellness analysis."""
    high_risk_schools: List[str] = []
    schools_with_retention_concerns: List[str] = []
    system_risk_level: RiskLevel = RiskLevel.MEDIUM
    equity_concerns: List[str] = []
    stability_indicators: Dict[str, Any] = Field(default_factory=dict)


class DistrictNarratives(BaseModel):
    """LLM response format for district narratives."""
    district_strengths: List[str]
    district_needs: List[str]
    executive_summary: str
    recommended_pd_strategy: List[str]
    board_stories: List[BoardStory]
    celebration_opportunities: List[str]
    resource_priorities: List[str]


class DistrictAgent(BaseAgent):
    """
    Top-layer agent for aggregating school summaries into district-level analysis.
    
    Responsibilities:
    - Aggregate SchoolSummary objects across multiple schools
    - Perform cross-school domain comparisons and rankings
    - Identify system-wide PD needs and strategic opportunities
    - Generate board-ready stories with data-backed insights
    - Provide superintendent-level strategic recommendations
    - Produce structured DistrictSummary outputs
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        **kwargs
    ):
        super().__init__(llm_client=llm_client, **kwargs)
        
        # Domain groupings for strategic analysis
        self.domain_categories = {
            'instructional_core': ['I-A', 'I-B', 'I-C', 'I-D', 'I-E', 'I-F'],
            'classroom_environment': ['II-A', 'II-B', 'II-C', 'II-D', 'II-E'],
            'teaching_practice': ['III-A', 'III-B', 'III-C', 'III-D', 'III-E'],
            'professional_growth': ['IV-A', 'IV-B', 'IV-C', 'IV-D', 'IV-E', 'IV-F']
        }
        
        # PD topic mapping for district-level strategic planning
        self.strategic_pd_areas = {
            'I-A': 'Content Knowledge & Standards Alignment',
            'I-B': 'Student Development & Learning Sciences', 
            'I-C': 'Differentiation & Equity Practices',
            'I-D': 'Instructional Strategies & Methods',
            'I-E': 'Assessment & Data-Driven Instruction',
            'I-F': 'Educational Technology Integration',
            'II-A': 'Learning Culture & Climate',
            'II-B': 'Classroom Management Systems',
            'II-C': 'Procedures & Routines',
            'II-D': 'Student Behavior & SEL',
            'II-E': 'Physical Learning Environment',
            'III-A': 'Clear Communication & Expectations',
            'III-B': 'Questioning & Discourse',
            'III-C': 'Student Engagement & Motivation',
            'III-D': 'Assessment in Instruction',
            'III-E': 'Responsive Teaching & Flexibility',
            'IV-A': 'Professional Reflection & Growth',
            'IV-B': 'Data Analysis & Decision Making',
            'IV-C': 'Family & Community Engagement',
            'IV-D': 'Professional Learning & Development',
            'IV-E': 'Collaboration & School Culture',
            'IV-F': 'Ethics & Professional Responsibilities'
        }
    
    @property
    def agent_type(self) -> str:
        return "DistrictAgent"
    
    @property
    def role_description(self) -> str:
        return "Aggregate school summaries into district-level strategic analysis with cross-school comparisons, system PD strategy, and board-ready insights"
    
    async def execute(self, district_input: DistrictInput) -> AgentResult:
        """
        Main execution method for processing district school summaries.
        
        Args:
            district_input: Input containing school summaries and configuration
            
        Returns:
            AgentResult with DistrictSummary or error information
        """
        try:
            if not district_input.school_summaries:
                return AgentResult(
                    success=False,
                    error="No school summaries provided for district analysis",
                    agent_id=self.agent_id,
                    execution_time_ms=0.0
                )
            
            self.logger.info(f"Processing {len(district_input.school_summaries)} schools for {district_input.organization_name}")
            
            # Phase 1: Compute cross-school domain statistics
            domain_metrics = self._compute_district_domain_statistics(district_input.school_summaries)
            
            # Phase 2: Analyze system-wide risk and equity factors
            risk_analysis = self._analyze_district_risk(district_input.school_summaries)
            
            # Phase 3: Generate school rankings and performance comparisons
            school_rankings = self._generate_school_rankings(
                district_input.school_summaries,
                domain_metrics,
                district_input.min_school_count_for_ranking
            )
            
            # Phase 4: Analyze district-wide PD needs and strategies
            pd_analysis = self._analyze_system_pd_needs(district_input.school_summaries)
            
            # Phase 5: Identify strategic priorities and opportunities
            strategic_priorities = self._identify_district_priorities(
                domain_metrics, 
                pd_analysis, 
                risk_analysis
            )
            
            # Phase 6: Classify schools for support and recognition
            school_classifications = self._classify_schools(
                district_input.school_summaries,
                domain_metrics,
                risk_analysis
            )
            
            # Phase 7: LLM synthesis for board stories and strategic narratives
            narratives = await self._synthesize_district_narratives(
                district_input,
                domain_metrics,
                risk_analysis,
                pd_analysis,
                strategic_priorities,
                school_classifications
            )
            
            # Phase 8: Build comprehensive district summary
            district_summary = self._build_district_summary(
                district_input,
                domain_metrics,
                risk_analysis,
                school_rankings,
                pd_analysis,
                strategic_priorities,
                school_classifications,
                narratives
            )
            
            self.logger.info(
                f"Successfully processed district {district_input.organization_name}",
                extra={
                    "num_schools": len(district_input.school_summaries),
                    "total_teachers": sum(s.num_teachers_analyzed for s in district_input.school_summaries),
                    "domains_analyzed": len(domain_metrics),
                    "system_risk_level": risk_analysis.system_risk_level.value,
                    "board_stories_created": len(narratives.get('board_stories', [])),
                    "high_performing_schools": len(school_classifications.get('high_performing_schools', []))
                }
            )
            
            return AgentResult(
                success=True,
                data={"district_summary": district_summary.model_dump()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,  # Will be set by tracking
                metadata={
                    "organization_name": district_input.organization_name,
                    "schools_count": len(district_input.school_summaries),
                    "domains_analyzed": len(domain_metrics),
                    "total_teachers": sum(s.num_teachers_analyzed for s in district_input.school_summaries)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process district analysis: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={"error_type": type(e).__name__}
            )
    
    def _compute_district_domain_statistics(self, school_summaries: List[SchoolSummary]) -> Dict[str, DomainDistrictMetrics]:
        """
        Compute domain-level statistics across all schools in the district.
        
        This method performs pure mathematical aggregation without LLM calls.
        """
        domain_metrics = {}
        
        # Collect all domain data across schools
        domain_data = defaultdict(lambda: {
            'school_stats': {},  # school_name -> {green: X, yellow: Y, red: Z}
            'school_percentages': {},  # school_name -> {green: X%, yellow: Y%, red: Z%}
            'total_teachers_by_school': {},  # school_name -> teacher count
            'all_scores': [],
            'performance_counts': Counter()
        })
        
        # Aggregate data across all schools
        for school_summary in school_summaries:
            school_name = school_summary.school_name
            
            for domain_id, domain_stats in school_summary.domain_stats.items():
                data = domain_data[domain_id]
                
                # Store school-level statistics
                data['school_stats'][school_name] = domain_stats.copy()
                
                # Get total teachers for this school in this domain
                total_teachers_domain = sum(domain_stats.values())
                data['total_teachers_by_school'][school_name] = total_teachers_domain
                
                # Calculate percentages for this school
                if total_teachers_domain > 0:
                    school_percentages = {}
                    for status, count in domain_stats.items():
                        percentage = count / total_teachers_domain
                        school_percentages[status] = percentage
                        data['performance_counts'][status] += count
                    data['school_percentages'][school_name] = school_percentages
                
                    # Estimate average score for this school based on distribution
                    # Green = 3.5, Yellow = 2.5, Red = 1.5 (rough estimates)
                    weighted_score = (
                        school_percentages.get(DomainStatus.GREEN, 0) * 3.5 +
                        school_percentages.get(DomainStatus.YELLOW, 0) * 2.5 +
                        school_percentages.get(DomainStatus.RED, 0) * 1.5
                    )
                    data['all_scores'].append(weighted_score)
        
        # Compute metrics for each domain
        for domain_id, data in domain_data.items():
            metrics = DomainDistrictMetrics(domain_id=domain_id)
            
            # School averages (estimated from status distribution)
            for school_name, percentages in data['school_percentages'].items():
                school_avg = (
                    percentages.get(DomainStatus.GREEN, 0) * 3.5 +
                    percentages.get(DomainStatus.YELLOW, 0) * 2.5 +
                    percentages.get(DomainStatus.RED, 0) * 1.5
                )
                metrics.school_averages[school_name] = school_avg
            
            # District average
            if data['all_scores']:
                metrics.district_average = statistics.mean(data['all_scores'])
            
            # Performance distribution across all schools
            metrics.performance_distribution = dict(data['performance_counts'])
            
            # Count schools and teachers analyzed
            metrics.total_schools_analyzed = len(data['school_stats'])
            metrics.total_teachers_analyzed = sum(data['total_teachers_by_school'].values())
            
            # Identify top performing and struggling schools
            school_scores = list(metrics.school_averages.items())
            if len(school_scores) >= 2:
                school_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Top performers: above average + top quartile
                if metrics.district_average:
                    top_threshold = max(metrics.district_average * 1.1, 3.0)
                    metrics.top_performing_schools = [
                        school for school, score in school_scores 
                        if score >= top_threshold
                    ]
                    
                    # Schools needing support: below average or below 2.5
                    support_threshold = min(metrics.district_average * 0.9, 2.5)
                    metrics.schools_needing_support = [
                        school for school, score in school_scores 
                        if score < support_threshold
                    ]
            
            domain_metrics[domain_id] = metrics
        
        return domain_metrics
    
    def _analyze_district_risk(self, school_summaries: List[SchoolSummary]) -> DistrictRiskAnalysis:
        """
        Analyze risk factors across the district.
        
        Performs deterministic aggregation of school-level risk signals.
        """
        risk_analysis = DistrictRiskAnalysis()
        
        school_risk_levels = []
        schools_with_retention_issues = []
        equity_flags = []
        
        # Analyze each school's risk profile
        for school_summary in school_summaries:
            school_name = school_summary.school_name
            school_risk = school_summary.school_risk_level
            school_risk_levels.append(school_risk)
            
            # High risk schools
            if school_risk == RiskLevel.HIGH:
                risk_analysis.high_risk_schools.append(school_name)
            
            # Retention concerns based on teacher support needs
            if (len(school_summary.teachers_needing_support) >= 
                school_summary.num_teachers_analyzed * 0.2):
                schools_with_retention_issues.append(school_name)
                risk_analysis.schools_with_retention_concerns.append(school_name)
            
            # Equity concerns - look for performance disparities
            # (This would be more sophisticated with demographic data)
            if (school_summary.overall_performance_level == DomainStatus.RED and
                school_summary.improvement_trend == TrendDirection.DECLINING):
                equity_flags.append(f"{school_name}: Declining performance with high need")
        
        # Calculate system-wide risk level
        total_schools = len(school_summaries)
        if total_schools > 0:
            high_risk_rate = len(risk_analysis.high_risk_schools) / total_schools
            retention_risk_rate = len(schools_with_retention_issues) / total_schools
            
            if high_risk_rate >= 0.25 or retention_risk_rate >= 0.3:
                risk_analysis.system_risk_level = RiskLevel.HIGH
            elif high_risk_rate >= 0.15 or retention_risk_rate >= 0.2:
                risk_analysis.system_risk_level = RiskLevel.MEDIUM
            else:
                risk_analysis.system_risk_level = RiskLevel.LOW
        
        # Set equity concerns
        risk_analysis.equity_concerns = equity_flags
        
        # Stability indicators
        stable_schools = sum(1 for s in school_summaries if s.improvement_trend == TrendDirection.STABLE)
        improving_schools = sum(1 for s in school_summaries if s.improvement_trend == TrendDirection.IMPROVING)
        
        risk_analysis.stability_indicators = {
            "stable_schools": stable_schools,
            "improving_schools": improving_schools,
            "declining_schools": total_schools - stable_schools - improving_schools,
            "system_stability_rate": stable_schools / total_schools if total_schools > 0 else 0
        }
        
        return risk_analysis
    
    def _generate_school_rankings(
        self,
        school_summaries: List[SchoolSummary],
        domain_metrics: Dict[str, DomainDistrictMetrics],
        min_school_count: int = 3
    ) -> Dict[str, List[SchoolRanking]]:
        """
        Generate school rankings by domain and overall performance.
        """
        if len(school_summaries) < min_school_count:
            self.logger.info(f"Insufficient schools ({len(school_summaries)}) for meaningful rankings")
            return {}
        
        rankings_by_domain = {}
        
        # Create rankings for each domain
        for domain_id, metrics in domain_metrics.items():
            school_scores = list(metrics.school_averages.items())
            school_scores.sort(key=lambda x: x[1], reverse=True)
            
            domain_rankings = []
            for rank, (school_name, score) in enumerate(school_scores, 1):
                # Find the corresponding school summary
                school_summary = next(
                    (s for s in school_summaries if s.school_name == school_name), 
                    None
                )
                
                if school_summary:
                    # Determine standout and improvement areas based on domain performance
                    standout_areas = []
                    improvement_areas = []
                    
                    domain_stats = school_summary.domain_stats.get(domain_id, {})
                    total_teachers = sum(domain_stats.values()) if domain_stats else 0
                    
                    if total_teachers > 0:
                        green_rate = domain_stats.get(DomainStatus.GREEN, 0) / total_teachers
                        red_rate = domain_stats.get(DomainStatus.RED, 0) / total_teachers
                        
                        if green_rate >= 0.6:
                            standout_areas.append(f"Strong {self.strategic_pd_areas.get(domain_id, domain_id)}")
                        if red_rate >= 0.3:
                            improvement_areas.append(f"{self.strategic_pd_areas.get(domain_id, domain_id)} needs focus")
                    
                    ranking = SchoolRanking(
                        school_id=school_summary.school_id,
                        school_name=school_name,
                        domain_scores={domain_id: score},
                        overall_rank=rank,
                        standout_areas=standout_areas,
                        improvement_areas=improvement_areas
                    )
                    domain_rankings.append(ranking)
            
            rankings_by_domain[domain_id] = domain_rankings
        
        # Create overall rankings (average across all domains)
        overall_scores = {}
        for school_summary in school_summaries:
            school_name = school_summary.school_name
            school_domain_scores = []
            
            for domain_id, metrics in domain_metrics.items():
                if school_name in metrics.school_averages:
                    school_domain_scores.append(metrics.school_averages[school_name])
            
            if school_domain_scores:
                overall_scores[school_name] = statistics.mean(school_domain_scores)
        
        # Sort and create overall rankings
        overall_school_scores = list(overall_scores.items())
        overall_school_scores.sort(key=lambda x: x[1], reverse=True)
        
        overall_rankings = []
        for rank, (school_name, score) in enumerate(overall_school_scores, 1):
            school_summary = next(
                (s for s in school_summaries if s.school_name == school_name), 
                None
            )
            
            if school_summary:
                # Collect all domain scores for this school
                all_domain_scores = {}
                for domain_id, metrics in domain_metrics.items():
                    if school_name in metrics.school_averages:
                        all_domain_scores[domain_id] = metrics.school_averages[school_name]
                
                # Identify overall standout and improvement areas
                standout_areas = school_summary.school_strengths[:2] if school_summary.school_strengths else []
                improvement_areas = school_summary.school_needs[:2] if school_summary.school_needs else []
                
                ranking = SchoolRanking(
                    school_id=school_summary.school_id,
                    school_name=school_name,
                    domain_scores=all_domain_scores,
                    overall_rank=rank,
                    standout_areas=standout_areas,
                    improvement_areas=improvement_areas
                )
                overall_rankings.append(ranking)
        
        rankings_by_domain["overall"] = overall_rankings
        
        return rankings_by_domain
    
    def _analyze_system_pd_needs(self, school_summaries: List[SchoolSummary]) -> SystemPDAnalysis:
        """
        Analyze Professional Development needs across the district.
        
        Identifies common needs, shared priorities, and strategic opportunities.
        """
        pd_analysis = SystemPDAnalysis()
        
        # Aggregate PD needs across all schools
        domain_need_counts = Counter()
        school_priority_domains = defaultdict(list)
        
        for school_summary in school_summaries:
            school_name = school_summary.school_name
            
            # Count priority domains across all schools
            for domain in school_summary.priority_domains:
                domain_need_counts[domain] += 1
                school_priority_domains[domain].append(school_name)
        
        # Most common needs (domains needed by multiple schools)
        pd_analysis.most_common_needs = dict(domain_need_counts)
        
        # Shared priority domains (needed by 2+ schools)
        min_schools_for_shared = max(2, len(school_summaries) // 3)
        pd_analysis.shared_priority_domains = [
            domain for domain, count in domain_need_counts.items()
            if count >= min_schools_for_shared
        ]
        
        # School-specific needs (unique to individual schools)
        all_priority_domains = set(domain_need_counts.keys())
        for school_summary in school_summaries:
            school_name = school_summary.school_name
            unique_needs = [
                domain for domain in school_summary.priority_domains
                if domain_need_counts[domain] == 1
            ]
            if unique_needs:
                pd_analysis.school_specific_needs[school_name] = unique_needs
        
        # Generate recommended initiatives
        pd_analysis.recommended_initiatives = self._generate_pd_initiatives(
            pd_analysis.shared_priority_domains,
            pd_analysis.most_common_needs,
            len(school_summaries)
        )
        
        # Identify cross-school cohort opportunities
        pd_analysis.cohort_opportunities = self._identify_cohort_opportunities(
            school_summaries,
            pd_analysis.shared_priority_domains
        )
        
        return pd_analysis
    
    def _generate_pd_initiatives(
        self,
        shared_domains: List[str],
        domain_counts: Dict[str, int],
        total_schools: int
    ) -> List[str]:
        """Generate strategic PD initiatives based on district needs."""
        
        initiatives = []
        
        # High-impact initiatives for shared priority domains
        high_impact_domains = [d for d in shared_domains if domain_counts.get(d, 0) >= total_schools // 2]
        
        for domain in high_impact_domains[:3]:  # Top 3 high-impact domains
            pd_topic = self.strategic_pd_areas.get(domain, f"Domain {domain}")
            initiatives.append(f"District-wide {pd_topic} initiative")
        
        # Category-based initiatives
        category_needs = defaultdict(int)
        for domain, count in domain_counts.items():
            for category, domains in self.domain_categories.items():
                if domain in domains:
                    category_needs[category] += count
        
        # Add category initiatives for high-need areas
        if category_needs:
            top_category = max(category_needs.items(), key=lambda x: x[1])
            if top_category[1] >= total_schools:
                category_name = top_category[0].replace('_', ' ').title()
                initiatives.append(f"System-wide {category_name} professional learning series")
        
        # Leadership development if many schools need support
        if len(shared_domains) >= 4:
            initiatives.append("Instructional leadership development for school administrators")
        
        return initiatives[:5]  # Limit to top 5 strategic initiatives
    
    def _identify_cohort_opportunities(
        self,
        school_summaries: List[SchoolSummary],
        shared_domains: List[str]
    ) -> List[str]:
        """Identify opportunities for cross-school professional learning cohorts."""
        
        opportunities = []
        
        # Look for schools with similar PD needs that could collaborate
        for domain in shared_domains[:3]:  # Focus on top shared domains
            schools_needing_domain = [
                s.school_name for s in school_summaries 
                if domain in s.priority_domains
            ]
            
            if len(schools_needing_domain) >= 2:
                pd_topic = self.strategic_pd_areas.get(domain, f"Domain {domain}")
                opportunities.append(
                    f"Cross-school {pd_topic} cohort with {', '.join(schools_needing_domain[:3])}{'...' if len(schools_needing_domain) > 3 else ''}"
                )
        
        # Look for schools with exemplar teachers who could mentor others
        exemplar_schools = [
            s.school_name for s in school_summaries 
            if len(s.exemplar_teachers) >= 2
        ]
        
        if len(exemplar_schools) >= 2:
            opportunities.append(
                f"Teacher leadership cohort leveraging expertise from {', '.join(exemplar_schools[:2])}"
            )
        
        return opportunities
    
    def _identify_district_priorities(
        self,
        domain_metrics: Dict[str, DomainDistrictMetrics],
        pd_analysis: SystemPDAnalysis,
        risk_analysis: DistrictRiskAnalysis
    ) -> Dict[str, Any]:
        """Identify strategic priorities for the district."""
        
        priorities = {
            "instructional_priorities": [],
            "support_priorities": [],
            "strategic_opportunities": [],
            "immediate_actions": []
        }
        
        # Instructional priorities based on domain performance
        weak_domains = []
        for domain_id, metrics in domain_metrics.items():
            if metrics.district_average and metrics.district_average < 2.5:
                weak_domains.append((domain_id, metrics.district_average))
        
        weak_domains.sort(key=lambda x: x[1])  # Sort by weakest first
        priorities["instructional_priorities"] = [
            self.strategic_pd_areas.get(domain, domain) 
            for domain, _ in weak_domains[:3]
        ]
        
        # Support priorities based on risk analysis
        if risk_analysis.high_risk_schools:
            priorities["support_priorities"].append(
                f"Intensive support for {len(risk_analysis.high_risk_schools)} high-risk schools"
            )
        
        if risk_analysis.schools_with_retention_concerns:
            priorities["support_priorities"].append(
                f"Teacher retention initiatives in {len(risk_analysis.schools_with_retention_concerns)} schools"
            )
        
        if risk_analysis.equity_concerns:
            priorities["support_priorities"].append("Address identified equity concerns")
        
        # Strategic opportunities
        if pd_analysis.cohort_opportunities:
            priorities["strategic_opportunities"].extend(pd_analysis.cohort_opportunities[:2])
        
        if len(domain_metrics) > 10:  # Many domains analyzed
            priorities["strategic_opportunities"].append("Comprehensive curriculum alignment initiative")
        
        # Immediate actions for urgent needs
        if risk_analysis.system_risk_level == RiskLevel.HIGH:
            priorities["immediate_actions"].append("Emergency intervention protocol for struggling schools")
        
        if len(pd_analysis.shared_priority_domains) >= 5:
            priorities["immediate_actions"].append("Accelerated professional development deployment")
        
        return priorities
    
    def _classify_schools(
        self,
        school_summaries: List[SchoolSummary],
        domain_metrics: Dict[str, DomainDistrictMetrics],
        risk_analysis: DistrictRiskAnalysis
    ) -> Dict[str, List[str]]:
        """Classify schools for recognition and support."""
        
        classifications = {
            "high_performing_schools": [],
            "schools_needing_support": [],
            "stable_schools": [],
            "pilot_ready_schools": []
        }
        
        for school_summary in school_summaries:
            school_name = school_summary.school_name
            
            # High performing: green overall, many exemplars, low risk
            if (school_summary.overall_performance_level == DomainStatus.GREEN and
                school_summary.school_risk_level == RiskLevel.LOW and
                len(school_summary.exemplar_teachers) >= 2):
                classifications["high_performing_schools"].append(school_name)
                
                # Also potential pilot schools
                if school_summary.improvement_trend == TrendDirection.IMPROVING:
                    classifications["pilot_ready_schools"].append(school_name)
            
            # Schools needing support: red overall, high risk, or declining trend
            elif (school_summary.overall_performance_level == DomainStatus.RED or
                  school_summary.school_risk_level == RiskLevel.HIGH or
                  school_summary.improvement_trend == TrendDirection.DECLINING):
                classifications["schools_needing_support"].append(school_name)
            
            # Stable schools: everything else
            else:
                classifications["stable_schools"].append(school_name)
                
                # Stable schools with good leadership might be pilot-ready
                if (school_summary.improvement_trend == TrendDirection.STABLE and
                    len(school_summary.exemplar_teachers) >= 1):
                    classifications["pilot_ready_schools"].append(school_name)
        
        return classifications
    
    async def _synthesize_district_narratives(
        self,
        district_input: DistrictInput,
        domain_metrics: Dict[str, DomainDistrictMetrics],
        risk_analysis: DistrictRiskAnalysis,
        pd_analysis: SystemPDAnalysis,
        strategic_priorities: Dict[str, Any],
        school_classifications: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Use LLM to synthesize district-level narratives and board stories.
        
        Creates strategic narratives, board stories, and executive summary.
        """
        try:
            # Prepare context for LLM
            context = self._prepare_district_narrative_context(
                district_input, domain_metrics, risk_analysis, pd_analysis, 
                strategic_priorities, school_classifications
            )
            
            # Generate narratives using LLM
            narratives = await self.llm_call_with_template(
                template_name="district_narratives",
                template_variables={
                    "organization_name": district_input.organization_name,
                    "school_count": len(district_input.school_summaries),
                    "teacher_count": sum(s.num_teachers_analyzed for s in district_input.school_summaries),
                    "district_overview": context["district_overview"],
                    "performance_analysis": context["performance_analysis"],
                    "strategic_priorities": context["strategic_priorities"],
                    "risk_assessment": context["risk_assessment"],
                    "pd_strategy": context["pd_strategy"],
                    "school_highlights": context["school_highlights"]
                },
                response_format=DistrictNarratives
            )
            
            if isinstance(narratives, DistrictNarratives):
                return {
                    "district_strengths": narratives.district_strengths,
                    "district_needs": narratives.district_needs,
                    "executive_summary": narratives.executive_summary,
                    "recommended_pd_strategy": narratives.recommended_pd_strategy,
                    "board_stories": narratives.board_stories,
                    "celebration_opportunities": narratives.celebration_opportunities,
                    "resource_priorities": narratives.resource_priorities
                }
            
        except Exception as e:
            self.logger.warning(f"LLM district narrative synthesis failed: {e}")
        
        # Fallback: Generate narratives programmatically
        return self._generate_fallback_district_narratives(
            domain_metrics, risk_analysis, pd_analysis, strategic_priorities, school_classifications
        )
    
    def _prepare_district_narrative_context(
        self,
        district_input: DistrictInput,
        domain_metrics: Dict[str, DomainDistrictMetrics],
        risk_analysis: DistrictRiskAnalysis,
        pd_analysis: SystemPDAnalysis,
        strategic_priorities: Dict[str, Any],
        school_classifications: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Prepare structured context for district narrative generation."""
        
        total_schools = len(district_input.school_summaries)
        total_teachers = sum(s.num_teachers_analyzed for s in district_input.school_summaries)
        
        # District overview
        district_overview = f"District analysis covers {total_schools} schools with {total_teachers} teachers. "
        district_overview += f"System risk level: {risk_analysis.system_risk_level.value}. "
        district_overview += f"High performing schools: {len(school_classifications.get('high_performing_schools', []))}. "
        district_overview += f"Schools needing support: {len(school_classifications.get('schools_needing_support', []))}."
        
        # Performance analysis
        strong_domains = [d for d, m in domain_metrics.items() if m.district_average and m.district_average >= 3.0]
        weak_domains = [d for d, m in domain_metrics.items() if m.district_average and m.district_average < 2.5]
        
        performance_analysis = f"District strengths in: {', '.join(strong_domains[:3]) if strong_domains else 'No consistently strong areas'}. "
        performance_analysis += f"Areas needing focus: {', '.join(weak_domains[:3]) if weak_domains else 'Generally balanced performance'}."
        
        # Strategic priorities
        priorities_text = "Key priorities: "
        all_priorities = (
            strategic_priorities.get("instructional_priorities", [])[:2] +
            strategic_priorities.get("support_priorities", [])[:2]
        )
        priorities_text += ', '.join(all_priorities[:3]) if all_priorities else "Maintain current momentum"
        
        # Risk assessment
        risk_text = f"System stability: {risk_analysis.stability_indicators.get('stable_schools', 0)}/{total_schools} stable. "
        if risk_analysis.high_risk_schools:
            risk_text += f"High-risk schools requiring intervention: {', '.join(risk_analysis.high_risk_schools)}. "
        if risk_analysis.equity_concerns:
            risk_text += f"Equity concerns identified in {len(risk_analysis.equity_concerns)} areas."
        
        # PD strategy
        pd_text = f"Shared PD priorities across {len(pd_analysis.shared_priority_domains)} domains. "
        if pd_analysis.cohort_opportunities:
            pd_text += f"Cross-school collaboration opportunities: {len(pd_analysis.cohort_opportunities)}. "
        pd_text += f"Recommended initiatives: {', '.join(pd_analysis.recommended_initiatives[:2])}"
        
        # School highlights
        highlights = ""
        if school_classifications.get("high_performing_schools"):
            highlights += f"Exemplary schools: {', '.join(school_classifications['high_performing_schools'][:3])}. "
        if school_classifications.get("pilot_ready_schools"):
            highlights += f"Innovation-ready schools: {', '.join(school_classifications['pilot_ready_schools'][:3])}."
        
        return {
            "district_overview": district_overview,
            "performance_analysis": performance_analysis,
            "strategic_priorities": priorities_text,
            "risk_assessment": risk_text,
            "pd_strategy": pd_text,
            "school_highlights": highlights or "All schools working toward continuous improvement"
        }
    
    def _generate_fallback_district_narratives(
        self,
        domain_metrics: Dict[str, DomainDistrictMetrics],
        risk_analysis: DistrictRiskAnalysis,
        pd_analysis: SystemPDAnalysis,
        strategic_priorities: Dict[str, Any],
        school_classifications: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Generate basic district narratives when LLM synthesis fails."""
        
        # District strengths
        district_strengths = []
        strong_domains = [d for d, m in domain_metrics.items() if m.district_average and m.district_average >= 3.0]
        if strong_domains:
            district_strengths.append(f"System-wide strength in {', '.join(strong_domains[:2])}")
        
        if len(school_classifications.get('high_performing_schools', [])) > 0:
            district_strengths.append(f"{len(school_classifications['high_performing_schools'])} schools demonstrating exceptional performance")
        
        if risk_analysis.system_risk_level == RiskLevel.LOW:
            district_strengths.append("Strong system stability with effective teacher retention")
        
        # District needs
        district_needs = []
        weak_domains = [d for d, m in domain_metrics.items() if m.district_average and m.district_average < 2.5]
        if weak_domains:
            district_needs.append(f"System-wide improvement needed in {', '.join(weak_domains[:2])}")
        
        if len(school_classifications.get('schools_needing_support', [])) > 0:
            district_needs.append(f"Targeted support for {len(school_classifications['schools_needing_support'])} schools")
        
        if risk_analysis.system_risk_level != RiskLevel.LOW:
            district_needs.append("Teacher wellness and retention initiatives")
        
        # Executive summary
        summary_parts = []
        total_schools = len(domain_metrics)
        if strong_domains and risk_analysis.system_risk_level == RiskLevel.LOW:
            summary_parts.append("District demonstrates strong overall performance with effective systems")
        elif weak_domains or risk_analysis.system_risk_level == RiskLevel.HIGH:
            summary_parts.append("District requires strategic focus in key areas with targeted interventions")
        else:
            summary_parts.append("District shows balanced performance with opportunities for strategic growth")
        
        summary_parts.append(f"Clear action plan in place across {total_schools} schools")
        executive_summary = ". ".join(summary_parts) + "."
        
        # Recommended PD strategy
        pd_strategy = pd_analysis.recommended_initiatives[:3] if pd_analysis.recommended_initiatives else [
            "Continue supporting instructional excellence across all schools",
            "Focus professional development on identified priority domains",
            "Strengthen collaboration between schools"
        ]
        
        # Board stories (simplified)
        board_stories = []
        if school_classifications.get('high_performing_schools'):
            board_stories.append(BoardStory(
                title="School Excellence Recognition",
                narrative=f"Celebrating outstanding performance at {', '.join(school_classifications['high_performing_schools'][:2])}",
                story_type="positive",
                supporting_data={"high_performing_count": len(school_classifications['high_performing_schools'])}
            ))
        
        if strategic_priorities.get('instructional_priorities'):
            board_stories.append(BoardStory(
                title="Strategic Instructional Focus",
                narrative=f"District-wide initiative targeting {strategic_priorities['instructional_priorities'][0]}",
                story_type="neutral",
                call_to_action="Approve professional development resource allocation"
            ))
        
        # Celebration opportunities
        celebration_opportunities = []
        if school_classifications.get('high_performing_schools'):
            celebration_opportunities.append("Recognize exemplary schools at board meeting")
        if pd_analysis.cohort_opportunities:
            celebration_opportunities.append("Highlight cross-school collaboration successes")
        
        # Resource priorities
        resource_priorities = []
        if strategic_priorities.get('support_priorities'):
            resource_priorities.extend(strategic_priorities['support_priorities'][:2])
        if strategic_priorities.get('instructional_priorities'):
            resource_priorities.append(f"Professional development in {strategic_priorities['instructional_priorities'][0]}")
        
        return {
            "district_strengths": district_strengths or ["Committed school teams working toward excellence"],
            "district_needs": district_needs or ["Continued focus on instructional improvement"],
            "executive_summary": executive_summary,
            "recommended_pd_strategy": pd_strategy,
            "board_stories": board_stories,
            "celebration_opportunities": celebration_opportunities or ["Celebrate ongoing progress across all schools"],
            "resource_priorities": resource_priorities or ["Support ongoing school improvement efforts"]
        }
    
    def _build_district_summary(
        self,
        district_input: DistrictInput,
        domain_metrics: Dict[str, DomainDistrictMetrics],
        risk_analysis: DistrictRiskAnalysis,
        school_rankings: Dict[str, List[SchoolRanking]],
        pd_analysis: SystemPDAnalysis,
        strategic_priorities: Dict[str, Any],
        school_classifications: Dict[str, List[str]],
        narratives: Dict[str, Any]
    ) -> DistrictSummary:
        """Build the final DistrictSummary output."""
        
        total_teachers = sum(s.num_teachers_analyzed for s in district_input.school_summaries)
        
        # Determine overall district health
        strong_domains = [d for d, m in domain_metrics.items() if m.district_average and m.district_average >= 3.0]
        weak_domains = [d for d, m in domain_metrics.items() if m.district_average and m.district_average < 2.5]
        total_domains = len(domain_metrics)
        
        if len(strong_domains) >= total_domains * 0.6 and len(weak_domains) <= total_domains * 0.2:
            overall_health = DomainStatus.GREEN
        elif len(weak_domains) >= total_domains * 0.4:
            overall_health = DomainStatus.RED
        else:
            overall_health = DomainStatus.YELLOW
        
        # Determine improvement momentum
        improving_schools = sum(1 for s in district_input.school_summaries if s.improvement_trend == TrendDirection.IMPROVING)
        declining_schools = sum(1 for s in district_input.school_summaries if s.improvement_trend == TrendDirection.DECLINING)
        
        if improving_schools > declining_schools and risk_analysis.system_risk_level == RiskLevel.LOW:
            momentum = TrendDirection.IMPROVING
        elif declining_schools > improving_schools or risk_analysis.system_risk_level == RiskLevel.HIGH:
            momentum = TrendDirection.DECLINING
        else:
            momentum = TrendDirection.STABLE
        
        # Common PD needs - aggregate from shared priorities and individual school data
        common_pd_needs = pd_analysis.most_common_needs.copy()
        
        return DistrictSummary(
            organization_id=district_input.organization_id,
            organization_name=district_input.organization_name,
            analysis_period_start=district_input.analysis_period_start,
            analysis_period_end=district_input.analysis_period_end,
            num_schools_analyzed=len(district_input.school_summaries),
            num_teachers_analyzed=total_teachers,
            
            # Priorities and focus
            priority_domains=pd_analysis.shared_priority_domains,
            district_focus_areas=strategic_priorities.get("instructional_priorities", []),
            
            # Performance overview
            district_strengths=narratives.get("district_strengths", []),
            district_needs=narratives.get("district_needs", []),
            
            # School analysis
            school_rankings_by_domain=school_rankings,
            high_performing_schools=school_classifications.get("high_performing_schools", []),
            schools_needing_support=school_classifications.get("schools_needing_support", []),
            
            # Board communication
            board_ready_stories=narratives.get("board_stories", []),
            executive_summary=narratives.get("executive_summary", "District analysis complete with strategic recommendations."),
            
            # Strategic recommendations
            recommended_PD_strategy=narratives.get("recommended_pd_strategy", []),
            pilot_opportunities=school_classifications.get("pilot_ready_schools", []),
            resource_allocation_priorities=narratives.get("resource_priorities", []),
            
            # System metrics
            overall_district_health=overall_health,
            system_risk_level=risk_analysis.system_risk_level,
            improvement_momentum=momentum,
            
            # Cross-cutting analysis
            common_PD_needs=common_pd_needs,
            equity_concerns=risk_analysis.equity_concerns,
            celebration_opportunities=narratives.get("celebration_opportunities", [])
        )