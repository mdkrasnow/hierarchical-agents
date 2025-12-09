"""
Utility functions for working with models and schemas.

Provides helper functions for:
- Converting database rows to Pydantic models
- Schema validation and parsing
- Performance threshold calculations
- Data transformation utilities
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from .database import DanielsonEvaluation, Organization, User
from .agent_outputs import DomainStatus, DomainSummary, TrendDirection


def parse_performance_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Union[int, float]]]:
    """Parse organization performance_level_config into domain thresholds."""
    if not config or "domains" not in config:
        # Return default thresholds if no config
        return {
            "default": {"green": 3, "yellow": 2, "red": 1}
        }
    
    domain_thresholds = {}
    for domain_id, domain_config in config["domains"].items():
        if isinstance(domain_config, dict):
            domain_thresholds[domain_id] = {
                "green": domain_config.get("green", 3),
                "yellow": domain_config.get("yellow", 2), 
                "red": domain_config.get("red", 1)
            }
    
    return domain_thresholds


def score_to_status(score: float, thresholds: Dict[str, Union[int, float]]) -> DomainStatus:
    """Convert numeric score to red/yellow/green status."""
    if score >= thresholds.get("green", 3):
        return DomainStatus.GREEN
    elif score >= thresholds.get("yellow", 2):
        return DomainStatus.YELLOW
    else:
        return DomainStatus.RED


def calculate_trend(scores: List[Tuple[datetime, float]], window_months: int = 6) -> TrendDirection:
    """Calculate trend direction from a list of (date, score) pairs."""
    if len(scores) < 2:
        return TrendDirection.UNKNOWN
    
    # Filter to recent window
    cutoff_date = datetime.now() - timedelta(days=window_months * 30)
    recent_scores = [(date, score) for date, score in scores if date >= cutoff_date]
    
    if len(recent_scores) < 2:
        return TrendDirection.STABLE
    
    # Sort by date
    recent_scores.sort(key=lambda x: x[0])
    
    # Simple linear trend calculation
    first_half = recent_scores[:len(recent_scores)//2]
    second_half = recent_scores[len(recent_scores)//2:]
    
    first_avg = sum(score for _, score in first_half) / len(first_half)
    second_avg = sum(score for _, score in second_half) / len(second_half)
    
    diff = second_avg - first_avg
    
    if diff > 0.2:
        return TrendDirection.IMPROVING
    elif diff < -0.2:
        return TrendDirection.DECLINING
    else:
        return TrendDirection.STABLE


def aggregate_domain_summaries(summaries: List[DomainSummary]) -> DomainSummary:
    """Aggregate multiple domain summaries into a single summary."""
    if not summaries:
        return DomainSummary(
            domain_id="unknown",
            status_color=DomainStatus.YELLOW,
            summary="No data available"
        )
    
    # Use the first summary as base
    base = summaries[0]
    
    # Aggregate scores
    scores = [s.score for s in summaries if s.score is not None]
    avg_score = sum(scores) / len(scores) if scores else None
    
    # Determine most recent status
    status_counts = {}
    for summary in summaries:
        status_counts[summary.status_color] = status_counts.get(summary.status_color, 0) + 1
    
    # Most common status (with red taking precedence)
    if DomainStatus.RED in status_counts:
        aggregated_status = DomainStatus.RED
    elif DomainStatus.YELLOW in status_counts:
        aggregated_status = DomainStatus.YELLOW
    else:
        aggregated_status = DomainStatus.GREEN
    
    # Combine evidence
    all_evidence = []
    all_growth = []
    all_concerns = []
    
    for summary in summaries:
        all_evidence.extend(summary.evidence_quotes)
        all_growth.extend(summary.growth_signals)
        all_concerns.extend(summary.concern_signals)
    
    # Deduplicate and limit
    unique_evidence = list(dict.fromkeys(all_evidence))[:3]  # Top 3 unique quotes
    unique_growth = list(dict.fromkeys(all_growth))[:3]
    unique_concerns = list(dict.fromkeys(all_concerns))[:3]
    
    return DomainSummary(
        domain_id=base.domain_id,
        score=avg_score,
        status_color=aggregated_status,
        summary=f"Aggregated from {len(summaries)} evaluations",
        growth_signals=unique_growth,
        concern_signals=unique_concerns,
        evidence_quotes=unique_evidence
    )


def extract_teacher_school_mapping(evaluations: List[DanielsonEvaluation]) -> Dict[str, List[str]]:
    """Extract mapping of teachers to schools from evaluation data."""
    mapping = {}
    
    for eval in evaluations:
        teacher = eval.teacher_name
        school = eval.school_name
        
        if teacher and school:
            if teacher not in mapping:
                mapping[teacher] = []
            if school not in mapping[teacher]:
                mapping[teacher].append(school)
    
    return mapping


def validate_agent_output_schema(output: Dict[str, Any], schema_type: str) -> Tuple[bool, List[str]]:
    """Validate agent output against expected schema."""
    errors = []
    
    if schema_type == "evaluation_summary":
        required_fields = ["teacher_name", "school_name", "evaluation_id", "date", "per_domain"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        if "per_domain" in output and not isinstance(output["per_domain"], dict):
            errors.append("per_domain must be a dictionary")
    
    elif schema_type == "teacher_summary":
        required_fields = ["teacher_name", "school_name", "overall_short_summary"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
    
    elif schema_type == "school_summary":
        required_fields = ["school_name", "domain_stats", "school_strengths"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
    
    elif schema_type == "district_summary":
        required_fields = ["organization_name", "priority_domains", "executive_summary"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors


def get_default_performance_config() -> Dict[str, Any]:
    """Get default performance configuration for new organizations."""
    return {
        "version": "1.0",
        "framework": "danielson_2023",
        "domains": {
            "I-A": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "I-B": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "I-C": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "II-A": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "II-B": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "III-A": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "III-B": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "III-C": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0},
            "IV-A": {"green": 3, "yellow": 2, "red": 1, "weight": 1.0}
        },
        "global_thresholds": {
            "exemplar_teacher": 3.5,
            "proficient_teacher": 2.8,
            "developing_teacher": 2.0
        }
    }


def format_domain_stats_for_display(
    stats: Dict[str, Dict[DomainStatus, int]]
) -> Dict[str, Dict[str, str]]:
    """Format domain stats for human-readable display."""
    formatted = {}
    
    for domain, counts in stats.items():
        total = sum(counts.values())
        if total == 0:
            continue
            
        formatted[domain] = {
            "total": str(total),
            "green": f"{counts.get(DomainStatus.GREEN, 0)} ({counts.get(DomainStatus.GREEN, 0)/total*100:.1f}%)",
            "yellow": f"{counts.get(DomainStatus.YELLOW, 0)} ({counts.get(DomainStatus.YELLOW, 0)/total*100:.1f}%)",
            "red": f"{counts.get(DomainStatus.RED, 0)} ({counts.get(DomainStatus.RED, 0)/total*100:.1f}%)"
        }
    
    return formatted