"""
Hierarchical agent JSON output schemas for the 4-level hierarchy:
District/ESC → School → Teacher → Evaluation

These schemas define the structured outputs that each agent level produces
and that get aggregated up the hierarchy.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class DomainStatus(str, Enum):
    """Red/Yellow/Green status for domains."""
    GREEN = "green"
    YELLOW = "yellow" 
    RED = "red"


class RiskLevel(str, Enum):
    """Risk levels for teachers/schools."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendDirection(str, Enum):
    """Trend direction indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


class DomainSummary(BaseModel):
    """Summary for a single domain within an evaluation/teacher/school."""
    domain_id: str  # e.g., "I-A", "II-B"
    score: Optional[float] = None
    status_color: DomainStatus
    trend: Optional[TrendDirection] = None
    summary: Optional[str] = None  # Brief narrative
    growth_signals: List[str] = []
    concern_signals: List[str] = []
    evidence_quotes: List[str] = []


class EvaluationSummary(BaseModel):
    """Output schema for EvaluationAgent (bottom layer)."""
    # Identity
    teacher_id: Optional[UUID] = None  # Might not have structured ID
    teacher_name: str
    school_id: Optional[UUID] = None
    school_name: str
    evaluation_id: UUID
    date: datetime
    
    # Domain analysis
    per_domain: Dict[str, DomainSummary] = {}
    
    # Flags and indicators
    flags: Dict[str, Any] = {
        "needs_PD": [],  # List of domain IDs
        "exemplar": False,
        "risk_of_leaving": False,
        "burnout_signals": False
    }
    
    # Evidence
    evidence_snippets: List[str] = []
    key_strengths: List[str] = []
    key_concerns: List[str] = []
    
    # Metadata
    relevance_to_question: str = "medium"  # high/medium/low
    evaluation_type: str = "formal"  # formal/informal

    @validator('per_domain')
    def validate_domains(cls, v):
        """Ensure domain summaries are properly structured."""
        for domain_id, summary in v.items():
            if not isinstance(summary, DomainSummary):
                raise ValueError(f"Domain {domain_id} summary must be DomainSummary instance")
        return v


class TeacherSummary(BaseModel):
    """Output schema for TeacherAgent."""
    # Identity
    teacher_id: Optional[UUID] = None
    teacher_name: str
    school_id: Optional[UUID] = None
    school_name: str
    
    # Time period
    evaluation_period_start: Optional[datetime] = None
    evaluation_period_end: Optional[datetime] = None
    num_evaluations: int = 0
    
    # Domain overview (aggregated across evaluations)
    per_domain_overview: Dict[str, DomainSummary] = {}
    
    # Recommendations and analysis
    recommended_PD_focus: List[str] = []  # Specific topic recommendations
    recommended_PD_domains: List[str] = []  # Domain IDs needing focus
    
    # Risk and performance
    risk_level: RiskLevel = RiskLevel.LOW
    overall_performance_trend: TrendDirection = TrendDirection.STABLE
    
    # Evidence and narrative
    notable_evidence: List[str] = []  # Up to 3-5 key quotes
    growth_story: Optional[str] = None  # Positive narrative
    concern_story: Optional[str] = None  # Areas needing attention
    overall_short_summary: str  # 2-3 sentences for quick view
    
    # Flags
    is_exemplar: bool = False
    needs_immediate_support: bool = False
    
    # Stats
    domain_distribution: Dict[DomainStatus, int] = {
        DomainStatus.GREEN: 0,
        DomainStatus.YELLOW: 0, 
        DomainStatus.RED: 0
    }

    @validator('overall_short_summary')
    def validate_summary_length(cls, v):
        """Ensure summary is concise."""
        if len(v.split()) > 50:  # Rough word count check
            raise ValueError("Summary should be 2-3 sentences (under 50 words)")
        return v


class PDCohort(BaseModel):
    """Professional Development cohort grouping."""
    domain_id: str
    focus_area: str  # e.g., "Questioning strategies"
    teacher_ids: List[UUID] = []
    teacher_names: List[str] = []  # For display when IDs not available
    priority_level: str = "medium"  # high/medium/low
    suggested_duration: Optional[str] = None  # "3 sessions", "ongoing"


class SchoolSummary(BaseModel):
    """Output schema for SchoolAgent."""
    # Identity
    school_id: Optional[UUID] = None
    school_name: str
    organization_id: Optional[UUID] = None
    
    # Time period and scope
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None
    num_teachers_analyzed: int = 0
    
    # Domain statistics
    domain_stats: Dict[str, Dict[DomainStatus, int]] = {}  # domain -> {green: X, yellow: Y, red: Z}
    domain_percentages: Dict[str, Dict[DomainStatus, float]] = {}
    
    # Professional Development planning
    PD_cohorts: List[PDCohort] = []
    priority_domains: List[str] = []  # Ordered by need
    
    # School-level narrative
    school_strengths: List[str] = []  # 2-3 bullet points
    school_needs: List[str] = []  # 2-3 bullet points
    
    # Stories for communication
    stories_for_principal: List[str] = []
    stories_for_supervisor_or_board: List[str] = []
    
    # Teacher identification
    exemplar_teachers: List[str] = []  # Names for recognition
    teachers_needing_support: List[str] = []  # Names for intervention
    
    # Overall metrics
    overall_performance_level: DomainStatus = DomainStatus.YELLOW
    school_risk_level: RiskLevel = RiskLevel.MEDIUM
    improvement_trend: TrendDirection = TrendDirection.STABLE

    @validator('domain_stats')
    def validate_domain_stats(cls, v):
        """Ensure stats are properly structured."""
        for domain, stats in v.items():
            if not all(status in stats for status in DomainStatus):
                # Fill missing statuses with 0
                for status in DomainStatus:
                    stats.setdefault(status, 0)
        return v


class SchoolRanking(BaseModel):
    """School ranking for district-level analysis."""
    school_id: Optional[UUID] = None
    school_name: str
    domain_scores: Dict[str, float] = {}  # domain -> average score
    overall_rank: Optional[int] = None
    standout_areas: List[str] = []  # What they excel at
    improvement_areas: List[str] = []  # What needs work


class BoardStory(BaseModel):
    """Board-ready narrative with supporting data."""
    title: str
    narrative: str  # 2-3 sentences
    supporting_data: Dict[str, Any] = {}  # Charts, numbers, etc.
    story_type: str = "neutral"  # positive/concern/neutral
    call_to_action: Optional[str] = None


class DistrictSummary(BaseModel):
    """Output schema for District/ESC Agent (top layer)."""
    # Identity and scope
    organization_id: UUID
    organization_name: str
    analysis_period_start: Optional[datetime] = None
    analysis_period_end: Optional[datetime] = None
    num_schools_analyzed: int = 0
    num_teachers_analyzed: int = 0
    
    # System-wide priorities
    priority_domains: List[str] = []  # Ordered by system-wide need
    district_focus_areas: List[str] = []  # High-level strategic areas
    
    # Performance overview
    district_strengths: List[str] = []  # 2-3 bullets
    district_needs: List[str] = []  # 2-3 bullets
    
    # School analysis
    school_rankings_by_domain: Dict[str, List[SchoolRanking]] = {}
    high_performing_schools: List[str] = []
    schools_needing_support: List[str] = []
    
    # Board communication
    board_ready_stories: List[BoardStory] = []
    executive_summary: str  # 1-2 paragraphs for leadership
    
    # Strategic recommendations
    recommended_PD_strategy: List[str] = []  # 1-3 actionable system moves
    pilot_opportunities: List[str] = []  # Where to test new programs
    resource_allocation_priorities: List[str] = []
    
    # System metrics
    overall_district_health: DomainStatus = DomainStatus.YELLOW
    system_risk_level: RiskLevel = RiskLevel.MEDIUM
    improvement_momentum: TrendDirection = TrendDirection.STABLE
    
    # Cross-cutting analysis
    common_PD_needs: Dict[str, int] = {}  # PD topic -> number of teachers
    equity_concerns: List[str] = []  # Disparities between schools
    celebration_opportunities: List[str] = []  # Success stories to highlight

    @validator('executive_summary')
    def validate_executive_summary_length(cls, v):
        """Ensure executive summary is appropriately sized."""
        word_count = len(v.split())
        if word_count > 150:
            raise ValueError("Executive summary should be 1-2 paragraphs (under 150 words)")
        return v

    @validator('board_ready_stories')
    def validate_board_stories(cls, v):
        """Ensure we have balance of story types."""
        if len(v) > 6:
            raise ValueError("Should have 2-6 board stories for digestibility")
        return v