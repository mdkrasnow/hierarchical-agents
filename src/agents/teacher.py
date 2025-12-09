"""
Teacher agent for aggregating and synthesizing evaluation summaries.

This module implements the second layer of the hierarchical agent system,
consuming EvaluationSummary outputs from EvaluationAgent and producing 
TeacherSummary outputs with professional development recommendations.
"""

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agents.base import BaseAgent, AgentResult
from models import EvaluationSummary, TeacherSummary, DomainSummary, DomainStatus, RiskLevel, TrendDirection
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


class TeacherInput(BaseModel):
    """Input data for teacher analysis."""
    evaluations: List[EvaluationSummary]
    teacher_id: Optional[UUID] = None
    teacher_name: str
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None
    organization_config: Dict[str, Any] = Field(default_factory=dict)
    pd_focus_limit: int = 3


class DomainMetrics(BaseModel):
    """Computed metrics for a single domain across evaluations."""
    domain_id: str
    scores: List[float] = []
    average_score: Optional[float] = None
    score_trend: TrendDirection = TrendDirection.UNKNOWN
    status_distribution: Dict[DomainStatus, int] = Field(default_factory=lambda: {
        DomainStatus.GREEN: 0,
        DomainStatus.YELLOW: 0,
        DomainStatus.RED: 0
    })
    growth_signals: List[str] = []
    concern_signals: List[str] = []
    evidence_quotes: List[str] = []


class RiskAnalysis(BaseModel):
    """Risk analysis aggregated across evaluations."""
    burnout_indicators: List[str] = []
    leaving_risk_factors: List[str] = []
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_trend: TrendDirection = TrendDirection.STABLE
    needs_immediate_support: bool = False


class PDRecommendations(BaseModel):
    """Professional development recommendations."""
    priority_domains: List[str] = []
    specific_topics: List[str] = []
    rationale: str = ""
    urgency_level: str = "medium"


class TeacherAgent(BaseAgent):
    """
    Second-layer agent for aggregating teacher evaluations into comprehensive profiles.
    
    Responsibilities:
    - Aggregate EvaluationSummary objects for one teacher
    - Compute deterministic domain metrics (averages, trends, distributions)
    - Analyze time-based changes and improvement patterns
    - Generate PD recommendations based on recurring weak domains
    - Combine risk flags across evaluations with trend analysis
    - Create evidence-backed narratives with LLM synthesis
    - Produce structured TeacherSummary outputs
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        **kwargs
    ):
        super().__init__(llm_client=llm_client, **kwargs)
        
        # Domain groupings for PD recommendations
        self.domain_groups = {
            'planning': ['I-A', 'I-B', 'I-C', 'I-D', 'I-E', 'I-F'],
            'environment': ['II-A', 'II-B', 'II-C', 'II-D', 'II-E'],
            'instruction': ['III-A', 'III-B', 'III-C', 'III-D', 'III-E'],
            'professionalism': ['IV-A', 'IV-B', 'IV-C', 'IV-D', 'IV-E', 'IV-F']
        }
        
        # PD topic mapping
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
        return "TeacherAgent"
    
    @property
    def role_description(self) -> str:
        return "Aggregate evaluation summaries into comprehensive teacher profiles with PD recommendations and evidence-backed narratives"
    
    async def execute(self, teacher_input: TeacherInput) -> AgentResult:
        """
        Main execution method for processing teacher evaluations.
        
        Args:
            teacher_input: Input containing evaluations and configuration
            
        Returns:
            AgentResult with TeacherSummary or error information
        """
        try:
            if not teacher_input.evaluations:
                return AgentResult(
                    success=False,
                    error="No evaluations provided for analysis",
                    agent_id=self.agent_id,
                    execution_time_ms=0.0
                )
            
            self.logger.info(f"Processing {len(teacher_input.evaluations)} evaluations for {teacher_input.teacher_name}")
            
            # Phase 1: Deterministic metric computation
            domain_metrics = self._compute_domain_metrics(teacher_input.evaluations)
            risk_analysis = self._analyze_risk_factors(teacher_input.evaluations)
            time_trends = self._analyze_time_trends(teacher_input.evaluations)
            
            # Phase 2: PD recommendation logic
            pd_recommendations = self._generate_pd_recommendations(
                domain_metrics, 
                teacher_input.organization_config,
                teacher_input.pd_focus_limit
            )
            
            # Phase 3: Evidence aggregation
            notable_evidence = self._aggregate_evidence(teacher_input.evaluations)
            
            # Phase 4: LLM synthesis for narratives
            narratives = await self._synthesize_narratives(
                teacher_input, 
                domain_metrics, 
                risk_analysis,
                pd_recommendations
            )
            
            # Phase 5: Build comprehensive teacher summary
            teacher_summary = self._build_teacher_summary(
                teacher_input,
                domain_metrics,
                risk_analysis,
                pd_recommendations,
                notable_evidence,
                narratives,
                time_trends
            )
            
            self.logger.info(
                f"Successfully processed teacher {teacher_input.teacher_name}",
                extra={
                    "num_evaluations": len(teacher_input.evaluations),
                    "domains_analyzed": len(domain_metrics),
                    "risk_level": risk_analysis.overall_risk_level.value,
                    "pd_recommendations": len(pd_recommendations.priority_domains)
                }
            )
            
            return AgentResult(
                success=True,
                data={"teacher_summary": teacher_summary.model_dump()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,  # Will be set by tracking
                metadata={
                    "teacher_name": teacher_input.teacher_name,
                    "evaluations_count": len(teacher_input.evaluations),
                    "domains_analyzed": len(domain_metrics)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process teacher analysis: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={"error_type": type(e).__name__}
            )
    
    def _compute_domain_metrics(self, evaluations: List[EvaluationSummary]) -> Dict[str, DomainMetrics]:
        """
        Compute deterministic metrics for each domain across evaluations.
        
        This method performs pure mathematical aggregation without LLM calls.
        """
        domain_metrics = {}
        
        # Collect all domain data
        domain_data = defaultdict(lambda: {
            'scores': [],
            'statuses': [],
            'growth_signals': [],
            'concern_signals': [],
            'evidence_quotes': []
        })
        
        # Aggregate data across evaluations
        for evaluation in evaluations:
            for domain_id, domain_summary in evaluation.per_domain.items():
                data = domain_data[domain_id]
                
                if domain_summary.score is not None:
                    data['scores'].append(domain_summary.score)
                
                data['statuses'].append(domain_summary.status_color)
                data['growth_signals'].extend(domain_summary.growth_signals)
                data['concern_signals'].extend(domain_summary.concern_signals)
                data['evidence_quotes'].extend(domain_summary.evidence_quotes)
        
        # Compute metrics for each domain
        for domain_id, data in domain_data.items():
            metrics = DomainMetrics(domain_id=domain_id)
            
            # Score statistics
            if data['scores']:
                metrics.scores = data['scores']
                metrics.average_score = statistics.mean(data['scores'])
                
                # Compute trend (simple: compare first half vs second half)
                if len(data['scores']) >= 4:
                    half = len(data['scores']) // 2
                    first_half_avg = statistics.mean(data['scores'][:half])
                    second_half_avg = statistics.mean(data['scores'][half:])
                    
                    if second_half_avg > first_half_avg + 0.2:
                        metrics.score_trend = TrendDirection.IMPROVING
                    elif second_half_avg < first_half_avg - 0.2:
                        metrics.score_trend = TrendDirection.DECLINING
                    else:
                        metrics.score_trend = TrendDirection.STABLE
            
            # Status distribution
            status_counts = Counter(data['statuses'])
            for status in DomainStatus:
                metrics.status_distribution[status] = status_counts.get(status, 0)
            
            # Aggregate signals (remove duplicates, limit)
            metrics.growth_signals = list(dict.fromkeys(data['growth_signals']))[:5]
            metrics.concern_signals = list(dict.fromkeys(data['concern_signals']))[:5]
            metrics.evidence_quotes = list(dict.fromkeys(data['evidence_quotes']))[:3]
            
            domain_metrics[domain_id] = metrics
        
        return domain_metrics
    
    def _analyze_risk_factors(self, evaluations: List[EvaluationSummary]) -> RiskAnalysis:
        """
        Analyze risk factors across all evaluations for this teacher.
        
        Performs deterministic aggregation of risk signals.
        """
        risk_analysis = RiskAnalysis()
        
        # Collect risk indicators
        burnout_count = 0
        leaving_risk_count = 0
        
        for evaluation in evaluations:
            flags = evaluation.flags or {}
            
            if flags.get('burnout_signals'):
                burnout_count += 1
                if hasattr(evaluation, 'burnout_indicators'):
                    risk_analysis.burnout_indicators.extend(evaluation.burnout_indicators)
            
            if flags.get('risk_of_leaving'):
                leaving_risk_count += 1
                if hasattr(evaluation, 'leaving_risk_factors'):
                    risk_analysis.leaving_risk_factors.extend(evaluation.leaving_risk_factors)
        
        # Remove duplicates
        risk_analysis.burnout_indicators = list(dict.fromkeys(risk_analysis.burnout_indicators))[:5]
        risk_analysis.leaving_risk_factors = list(dict.fromkeys(risk_analysis.leaving_risk_factors))[:5]
        
        # Determine overall risk level
        total_evaluations = len(evaluations)
        if total_evaluations > 0:
            burnout_rate = burnout_count / total_evaluations
            leaving_rate = leaving_risk_count / total_evaluations
            
            if leaving_rate >= 0.5 or burnout_rate >= 0.7:
                risk_analysis.overall_risk_level = RiskLevel.HIGH
                risk_analysis.needs_immediate_support = True
            elif leaving_rate >= 0.3 or burnout_rate >= 0.4:
                risk_analysis.overall_risk_level = RiskLevel.MEDIUM
            else:
                risk_analysis.overall_risk_level = RiskLevel.LOW
        
        # Analyze trend (simple: more recent evaluations show more risk)
        if len(evaluations) >= 3:
            recent_risk = sum(1 for eval in evaluations[-2:] 
                            if eval.flags.get('burnout_signals') or eval.flags.get('risk_of_leaving'))
            earlier_risk = sum(1 for eval in evaluations[:-2] 
                             if eval.flags.get('burnout_signals') or eval.flags.get('risk_of_leaving'))
            
            recent_rate = recent_risk / min(2, len(evaluations))
            earlier_rate = earlier_risk / max(1, len(evaluations) - 2)
            
            if recent_rate > earlier_rate + 0.2:
                risk_analysis.risk_trend = TrendDirection.DECLINING  # Increasing risk
            elif recent_rate < earlier_rate - 0.2:
                risk_analysis.risk_trend = TrendDirection.IMPROVING  # Decreasing risk
            else:
                risk_analysis.risk_trend = TrendDirection.STABLE
        
        return risk_analysis
    
    def _analyze_time_trends(self, evaluations: List[EvaluationSummary]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(evaluations) < 2:
            return {"overall_trend": TrendDirection.UNKNOWN, "trend_analysis": "Insufficient data"}
        
        # Sort evaluations by date
        sorted_evaluations = sorted(evaluations, key=lambda e: e.date)
        
        # Track domain improvements/declines
        domain_trends = {}
        overall_scores = []
        
        for evaluation in sorted_evaluations:
            # Calculate overall score for this evaluation
            domain_scores = [d.score for d in evaluation.per_domain.values() if d.score is not None]
            if domain_scores:
                overall_scores.append(statistics.mean(domain_scores))
        
        # Determine overall trend
        overall_trend = TrendDirection.STABLE
        if len(overall_scores) >= 2:
            trend_slope = (overall_scores[-1] - overall_scores[0]) / len(overall_scores)
            if trend_slope > 0.1:
                overall_trend = TrendDirection.IMPROVING
            elif trend_slope < -0.1:
                overall_trend = TrendDirection.DECLINING
        
        return {
            "overall_trend": overall_trend,
            "overall_scores": overall_scores,
            "num_evaluations": len(evaluations),
            "time_span_days": (sorted_evaluations[-1].date - sorted_evaluations[0].date).days
        }
    
    def _generate_pd_recommendations(
        self, 
        domain_metrics: Dict[str, DomainMetrics], 
        org_config: Dict[str, Any],
        focus_limit: int = 3
    ) -> PDRecommendations:
        """
        Generate professional development recommendations based on domain analysis.
        
        Pure logic-based recommendations without LLM calls.
        """
        recommendations = PDRecommendations()
        
        # Identify domains needing support
        priority_scores = {}
        
        for domain_id, metrics in domain_metrics.items():
            priority_score = 0
            
            # Factor 1: Low average score
            if metrics.average_score is not None:
                if metrics.average_score < 2.0:
                    priority_score += 3  # High priority
                elif metrics.average_score < 2.5:
                    priority_score += 2  # Medium priority
                elif metrics.average_score < 3.0:
                    priority_score += 1  # Low priority
            
            # Factor 2: High red/yellow status count
            total_evaluations = sum(metrics.status_distribution.values())
            if total_evaluations > 0:
                red_rate = metrics.status_distribution[DomainStatus.RED] / total_evaluations
                yellow_rate = metrics.status_distribution[DomainStatus.YELLOW] / total_evaluations
                
                if red_rate >= 0.5:
                    priority_score += 3
                elif red_rate >= 0.3:
                    priority_score += 2
                elif yellow_rate >= 0.7:
                    priority_score += 1
            
            # Factor 3: Declining trend
            if metrics.score_trend == TrendDirection.DECLINING:
                priority_score += 2
            
            # Factor 4: Recurring concerns
            if len(metrics.concern_signals) >= 3:
                priority_score += 1
            
            if priority_score > 0:
                priority_scores[domain_id] = priority_score
        
        # Select top priority domains
        sorted_domains = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations.priority_domains = [domain for domain, score in sorted_domains[:focus_limit]]
        
        # Map to specific PD topics
        recommendations.specific_topics = [
            self.pd_topics.get(domain_id, f"Domain {domain_id} Development")
            for domain_id in recommendations.priority_domains
        ]
        
        # Set urgency based on highest priority score
        if sorted_domains and sorted_domains[0][1] >= 6:
            recommendations.urgency_level = "high"
        elif sorted_domains and sorted_domains[0][1] >= 3:
            recommendations.urgency_level = "medium"
        else:
            recommendations.urgency_level = "low"
        
        # Create rationale
        if recommendations.priority_domains:
            domain_names = [self.pd_topics.get(d, d) for d in recommendations.priority_domains[:2]]
            recommendations.rationale = f"Focus needed in {', '.join(domain_names)} based on consistent performance gaps and evaluation patterns."
        else:
            recommendations.rationale = "Performance is generally strong across all domains. Consider advanced professional growth opportunities."
        
        return recommendations
    
    def _aggregate_evidence(self, evaluations: List[EvaluationSummary]) -> List[str]:
        """Aggregate and prioritize evidence snippets across evaluations."""
        all_evidence = []
        
        # Collect evidence from all evaluations
        for evaluation in evaluations:
            all_evidence.extend(evaluation.evidence_snippets)
            all_evidence.extend(evaluation.key_strengths)
        
        # Remove duplicates while preserving order
        unique_evidence = list(dict.fromkeys(all_evidence))
        
        # Prioritize evidence (simple heuristic: longer, more specific evidence first)
        prioritized_evidence = sorted(
            unique_evidence, 
            key=lambda x: (len(x), 'student' in x.lower(), 'learn' in x.lower()), 
            reverse=True
        )
        
        return prioritized_evidence[:5]  # Limit to top 5 pieces
    
    async def _synthesize_narratives(
        self,
        teacher_input: TeacherInput,
        domain_metrics: Dict[str, DomainMetrics],
        risk_analysis: RiskAnalysis,
        pd_recommendations: PDRecommendations
    ) -> Dict[str, str]:
        """
        Use LLM to synthesize evidence-backed narratives.
        
        Creates growth story, concern story, and overall summary.
        """
        try:
            # Prepare context for LLM
            context = self._prepare_narrative_context(
                teacher_input, domain_metrics, risk_analysis, pd_recommendations
            )
            
            # Generate narratives using LLM
            narratives = await self.llm_call_with_template(
                template_name="teacher_narratives",
                template_variables={
                    "teacher_name": teacher_input.teacher_name,
                    "evaluation_count": len(teacher_input.evaluations),
                    "domain_analysis": context["domain_analysis"],
                    "risk_analysis": context["risk_summary"],
                    "pd_recommendations": context["pd_summary"],
                    "evidence_highlights": context["evidence_summary"]
                },
                response_format=TeacherNarratives
            )
            
            if isinstance(narratives, TeacherNarratives):
                return {
                    "growth_story": narratives.growth_story,
                    "concern_story": narratives.concern_story,
                    "overall_summary": narratives.overall_summary
                }
            
        except Exception as e:
            self.logger.warning(f"LLM narrative synthesis failed: {e}")
        
        # Fallback: Generate narratives programmatically
        return self._generate_fallback_narratives(domain_metrics, risk_analysis, pd_recommendations)
    
    def _prepare_narrative_context(
        self,
        teacher_input: TeacherInput,
        domain_metrics: Dict[str, DomainMetrics],
        risk_analysis: RiskAnalysis,
        pd_recommendations: PDRecommendations
    ) -> Dict[str, str]:
        """Prepare structured context for narrative generation."""
        
        # Domain analysis summary
        strong_domains = [d for d, m in domain_metrics.items() if m.average_score and m.average_score >= 3.0]
        weak_domains = [d for d, m in domain_metrics.items() if m.average_score and m.average_score < 2.5]
        
        domain_analysis = f"Strong performance in: {', '.join(strong_domains[:3]) if strong_domains else 'No consistently strong domains'}. "
        domain_analysis += f"Areas needing support: {', '.join(weak_domains[:3]) if weak_domains else 'All domains performing adequately'}."
        
        # Risk summary
        risk_summary = f"Risk level: {risk_analysis.overall_risk_level.value}. "
        if risk_analysis.burnout_indicators:
            risk_summary += f"Burnout signals detected. "
        if risk_analysis.leaving_risk_factors:
            risk_summary += f"Retention concerns noted. "
        
        # PD summary
        pd_summary = f"Priority development areas: {', '.join(pd_recommendations.specific_topics[:2])}. "
        pd_summary += f"Urgency: {pd_recommendations.urgency_level}. {pd_recommendations.rationale}"
        
        # Evidence summary
        evidence_summary = "Key evidence includes teacher demonstrations of effective practices and areas for growth based on classroom observations."
        
        return {
            "domain_analysis": domain_analysis,
            "risk_summary": risk_summary,
            "pd_summary": pd_summary,
            "evidence_summary": evidence_summary
        }
    
    def _generate_fallback_narratives(
        self,
        domain_metrics: Dict[str, DomainMetrics],
        risk_analysis: RiskAnalysis,
        pd_recommendations: PDRecommendations
    ) -> Dict[str, str]:
        """Generate basic narratives when LLM synthesis fails."""
        
        # Growth story
        strong_domains = [d for d, m in domain_metrics.items() if m.average_score and m.average_score >= 3.0]
        growth_story = None
        if strong_domains:
            growth_story = f"Demonstrates consistent strength in {', '.join(strong_domains[:2])}, showing effective teaching practices and positive student outcomes."
        
        # Concern story  
        weak_domains = [d for d, m in domain_metrics.items() if m.average_score and m.average_score < 2.5]
        concern_story = None
        if weak_domains or risk_analysis.overall_risk_level != RiskLevel.LOW:
            concerns = []
            if weak_domains:
                concerns.append(f"needs development in {', '.join(weak_domains[:2])}")
            if risk_analysis.overall_risk_level == RiskLevel.HIGH:
                concerns.append("showing signs of stress or disengagement")
            concern_story = f"Teacher {', and '.join(concerns)}. Targeted support recommended."
        
        # Overall summary
        if strong_domains and not weak_domains:
            overall_summary = "Strong performer across evaluation domains with consistent positive outcomes."
        elif weak_domains and risk_analysis.overall_risk_level != RiskLevel.LOW:
            overall_summary = "Requires targeted professional development and support to address performance and wellness concerns."
        else:
            overall_summary = "Developing teacher with areas of strength and focused growth opportunities identified."
        
        return {
            "growth_story": growth_story,
            "concern_story": concern_story,
            "overall_summary": overall_summary
        }
    
    def _build_teacher_summary(
        self,
        teacher_input: TeacherInput,
        domain_metrics: Dict[str, DomainMetrics],
        risk_analysis: RiskAnalysis,
        pd_recommendations: PDRecommendations,
        notable_evidence: List[str],
        narratives: Dict[str, str],
        time_trends: Dict[str, Any]
    ) -> TeacherSummary:
        """Build the final TeacherSummary output."""
        
        # Build per-domain overview
        per_domain_overview = {}
        for domain_id, metrics in domain_metrics.items():
            # Determine overall status for this domain
            if metrics.average_score is not None:
                if metrics.average_score >= 3.0:
                    status_color = DomainStatus.GREEN
                elif metrics.average_score >= 2.0:
                    status_color = DomainStatus.YELLOW
                else:
                    status_color = DomainStatus.RED
            else:
                # Use most common status if no scores
                status_counts = metrics.status_distribution
                status_color = max(status_counts.items(), key=lambda x: x[1])[0]
            
            per_domain_overview[domain_id] = DomainSummary(
                domain_id=domain_id,
                score=metrics.average_score,
                status_color=status_color,
                trend=metrics.score_trend,
                summary=f"Average performance: {metrics.average_score:.1f}" if metrics.average_score else "Performance data available",
                growth_signals=metrics.growth_signals,
                concern_signals=metrics.concern_signals,
                evidence_quotes=metrics.evidence_quotes
            )
        
        # Calculate domain distribution
        domain_distribution = {
            DomainStatus.GREEN: 0,
            DomainStatus.YELLOW: 0,
            DomainStatus.RED: 0
        }
        
        for domain_summary in per_domain_overview.values():
            domain_distribution[domain_summary.status_color] += 1
        
        # Determine if exemplar teacher
        total_domains = len(per_domain_overview)
        is_exemplar = False
        if total_domains > 0:
            green_percentage = domain_distribution[DomainStatus.GREEN] / total_domains
            is_exemplar = green_percentage >= 0.8 and risk_analysis.overall_risk_level == RiskLevel.LOW
        
        return TeacherSummary(
            teacher_id=teacher_input.teacher_id,
            teacher_name=teacher_input.teacher_name,
            school_id=teacher_input.evaluations[0].school_id if teacher_input.evaluations else None,
            school_name=teacher_input.evaluations[0].school_name if teacher_input.evaluations else "Unknown School",
            evaluation_period_start=teacher_input.analysis_period_start or min(e.date for e in teacher_input.evaluations),
            evaluation_period_end=teacher_input.analysis_period_end or max(e.date for e in teacher_input.evaluations),
            num_evaluations=len(teacher_input.evaluations),
            per_domain_overview=per_domain_overview,
            recommended_PD_focus=pd_recommendations.specific_topics,
            recommended_PD_domains=pd_recommendations.priority_domains,
            risk_level=risk_analysis.overall_risk_level,
            overall_performance_trend=time_trends["overall_trend"],
            notable_evidence=notable_evidence,
            growth_story=narratives.get("growth_story"),
            concern_story=narratives.get("concern_story"),
            overall_short_summary=narratives["overall_summary"],
            is_exemplar=is_exemplar,
            needs_immediate_support=risk_analysis.needs_immediate_support,
            domain_distribution=domain_distribution
        )


class TeacherNarratives(BaseModel):
    """LLM response format for teacher narratives."""
    growth_story: Optional[str] = None
    concern_story: Optional[str] = None
    overall_summary: str