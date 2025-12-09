"""
Core evaluation agent for processing individual Danielson evaluations.

This module implements the bottom layer of the hierarchical agent system,
processing single evaluations to extract structured summaries with domain
analysis, risk flags, and evidence extraction.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agents.base import BaseAgent, AgentResult
from models import EvaluationSummary, DomainSummary, DomainStatus, DanielsonEvaluation
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


class EvaluationInput(BaseModel):
    """Input data for evaluation processing."""
    evaluation_data: DanielsonEvaluation
    organization_config: Dict[str, Any]
    analysis_focus: Optional[str] = "comprehensive"
    include_evidence: bool = True
    max_evidence_snippets: int = 10


class RiskSignals(BaseModel):
    """Structured risk signals extracted from evaluations."""
    burnout_indicators: List[str] = []
    disengagement_signals: List[str] = []
    leaving_risk_factors: List[str] = []
    confidence_score: float = 0.0


class DanielsonEvaluationAgent(BaseAgent):
    """
    Bottom-layer agent for processing individual Danielson evaluations.
    
    Responsibilities:
    - Extract domain scores and classify as red/yellow/green
    - Identify evidence snippets linked to specific domains
    - Detect risk flags (burnout, disengagement, leaving risk)
    - Handle missing data and partial evaluations gracefully
    - Produce structured EvaluationSummary outputs
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        **kwargs
    ):
        super().__init__(llm_client=llm_client, **kwargs)
        
        # Domain extraction patterns
        self.domain_patterns = {
            'danielson_domains': [
                'I-A', 'I-B', 'I-C', 'I-D', 'I-E', 'I-F',
                'II-A', 'II-B', 'II-C', 'II-D', 'II-E',
                'III-A', 'III-B', 'III-C', 'III-D', 'III-E',
                'IV-A', 'IV-B', 'IV-C', 'IV-D', 'IV-E', 'IV-F'
            ]
        }
        
        # Risk detection keywords
        self.risk_keywords = {
            'burnout': [
                'overwhelmed', 'exhausted', 'stress', 'burned out', 'tired',
                'struggling to keep up', 'feeling defeated', 'workload concerns',
                'work-life balance', 'emotional fatigue'
            ],
            'disengagement': [
                'going through motions', 'minimal effort', 'disinterested',
                'lack of enthusiasm', 'not participating', 'withdrawn',
                'avoiding responsibilities', 'decreased participation'
            ],
            'leaving_risk': [
                'considering leaving', 'looking elsewhere', 'updating resume',
                'retirement thoughts', 'career change', 'job searching',
                'exit interview', 'resignation'
            ]
        }
    
    @property
    def agent_type(self) -> str:
        return "DanielsonEvaluationAgent"
    
    @property
    def role_description(self) -> str:
        return "Process individual Danielson evaluations to extract structured domain analysis, evidence, and risk indicators"
    
    async def execute(self, evaluation_input: EvaluationInput) -> AgentResult:
        """
        Main execution method for processing a single evaluation.
        
        Args:
            evaluation_input: Input containing evaluation data and configuration
            
        Returns:
            AgentResult with EvaluationSummary or error information
        """
        try:
            self.logger.info(f"Processing evaluation {evaluation_input.evaluation_data.id}")
            
            # Extract basic metadata
            metadata = evaluation_input.evaluation_data.get_metadata()
            
            # Process domain scores
            domain_summaries = await self._process_domain_scores(
                evaluation_input.evaluation_data,
                evaluation_input.organization_config
            )
            
            # Extract evidence snippets
            evidence_snippets = []
            if evaluation_input.include_evidence:
                evidence_snippets = await self._extract_evidence_snippets(
                    evaluation_input.evaluation_data,
                    max_snippets=evaluation_input.max_evidence_snippets
                )
            
            # Detect risk signals
            risk_signals = await self._detect_risk_signals(
                evaluation_input.evaluation_data
            )
            
            # Generate flags and indicators
            flags = self._generate_flags(domain_summaries, risk_signals, evaluation_input.organization_config)
            
            # Extract key strengths and concerns
            strengths, concerns = await self._extract_strengths_and_concerns(
                evaluation_input.evaluation_data,
                domain_summaries
            )
            
            # Determine relevance and evaluation type
            relevance = self._assess_relevance(evaluation_input.evaluation_data)
            eval_type = "formal" if not metadata.is_informal else "informal"
            
            # Build evaluation summary
            summary = EvaluationSummary(
                teacher_id=None,  # Will be set if available in evaluation data
                teacher_name=metadata.teacher_name or "Unknown Teacher",
                school_id=None,   # Will be set if available 
                school_name=metadata.school_name or "Unknown School",
                evaluation_id=evaluation_input.evaluation_data.id,
                date=evaluation_input.evaluation_data.created_at,
                per_domain=domain_summaries,
                flags=flags,
                evidence_snippets=evidence_snippets,
                key_strengths=strengths,
                key_concerns=concerns,
                relevance_to_question=relevance,
                evaluation_type=eval_type
            )
            
            self.logger.info(
                f"Successfully processed evaluation {evaluation_input.evaluation_data.id}",
                extra={
                    "domains_processed": len(domain_summaries),
                    "evidence_count": len(evidence_snippets),
                    "risk_signals": len(risk_signals.burnout_indicators + risk_signals.disengagement_signals)
                }
            )
            
            return AgentResult(
                success=True,
                data={"evaluation_summary": summary.model_dump()},
                agent_id=self.agent_id,
                execution_time_ms=0.0,  # Will be set by tracking
                metadata={
                    "evaluation_id": str(evaluation_input.evaluation_data.id),
                    "domains_count": len(domain_summaries),
                    "evidence_count": len(evidence_snippets)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process evaluation: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=0.0,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _process_domain_scores(
        self,
        evaluation: DanielsonEvaluation,
        org_config: Dict[str, Any]
    ) -> Dict[str, DomainSummary]:
        """
        Extract and classify domain scores from evaluation data.
        
        Args:
            evaluation: The evaluation to process
            org_config: Organization configuration with thresholds
            
        Returns:
            Dictionary mapping domain IDs to DomainSummary objects
        """
        domain_summaries = {}
        
        try:
            # Get domain scores from evaluation
            domain_scores = evaluation.get_domain_scores()
            
            # Get thresholds from org config
            domain_configs = org_config.get('domains', {})
            
            for domain_score in domain_scores:
                domain_id = domain_score.domain_id
                
                # Get domain configuration
                domain_config = domain_configs.get(domain_id, {
                    'green': 3, 'yellow': 2, 'red': 1  # Default thresholds
                })
                
                # Determine status based on score
                status = self._classify_domain_score(
                    domain_score.overall_score,
                    domain_config
                )
                
                # Extract evidence from domain notes
                evidence_quotes = []
                if domain_score.notes:
                    evidence_quotes = self._extract_quotes_from_text(domain_score.notes)
                
                # Analyze for growth and concern signals
                growth_signals, concern_signals = self._analyze_domain_signals(
                    domain_score.notes or "",
                    domain_score.component_scores
                )
                
                domain_summary = DomainSummary(
                    domain_id=domain_id,
                    score=domain_score.overall_score,
                    status_color=status,
                    summary=self._generate_domain_summary(domain_score),
                    growth_signals=growth_signals,
                    concern_signals=concern_signals,
                    evidence_quotes=evidence_quotes[:3]  # Limit to top 3 quotes
                )
                
                domain_summaries[domain_id] = domain_summary
            
            # Handle missing domains with unknown status
            for domain_id in self.domain_patterns['danielson_domains']:
                if domain_id not in domain_summaries:
                    domain_summaries[domain_id] = DomainSummary(
                        domain_id=domain_id,
                        score=None,
                        status_color=DomainStatus.YELLOW,  # Default for missing
                        summary="No data available for this domain",
                        growth_signals=[],
                        concern_signals=["Missing evaluation data"],
                        evidence_quotes=[]
                    )
            
        except Exception as e:
            self.logger.warning(f"Error processing domain scores: {e}")
            # Return empty dict rather than failing completely
            
        return domain_summaries
    
    def _classify_domain_score(
        self,
        score: Optional[Union[int, float]],
        domain_config: Dict[str, Any]
    ) -> DomainStatus:
        """Classify a domain score as red/yellow/green based on thresholds."""
        if score is None:
            return DomainStatus.YELLOW  # Default for missing scores
        
        try:
            score = float(score)
            green_threshold = domain_config.get('green', 3)
            yellow_threshold = domain_config.get('yellow', 2)
            
            if score >= green_threshold:
                return DomainStatus.GREEN
            elif score >= yellow_threshold:
                return DomainStatus.YELLOW
            else:
                return DomainStatus.RED
                
        except (ValueError, TypeError):
            return DomainStatus.YELLOW  # Default for invalid scores
    
    def _extract_quotes_from_text(self, text: str) -> List[str]:
        """Extract meaningful quotes from evaluation text."""
        if not text or not text.strip():
            return []
        
        # Split into sentences and filter for substantive ones
        sentences = re.split(r'[.!?]+', text)
        quotes = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter for meaningful sentences (not too short, has substance)
            if (len(sentence) > 20 and 
                not sentence.lower().startswith(('the', 'a', 'an')) and
                any(word in sentence.lower() for word in [
                    'student', 'teach', 'learn', 'class', 'instruct', 'assess',
                    'engage', 'question', 'discuss', 'explain', 'demonstrate'
                ])):
                quotes.append(sentence)
                if len(quotes) >= 5:  # Limit quotes per domain
                    break
        
        return quotes
    
    def _analyze_domain_signals(
        self,
        notes: str,
        component_scores: Dict[str, Union[int, float]]
    ) -> Tuple[List[str], List[str]]:
        """Analyze domain notes and scores for growth and concern signals."""
        growth_signals = []
        concern_signals = []
        
        if not notes:
            if component_scores:
                concern_signals.append("Missing narrative notes")
            return growth_signals, concern_signals
        
        notes_lower = notes.lower()
        
        # Growth indicators
        growth_keywords = [
            'improve', 'growth', 'progress', 'develop', 'excel', 'strong',
            'effective', 'successful', 'innovative', 'creative', 'engaged'
        ]
        
        for keyword in growth_keywords:
            if keyword in notes_lower:
                # Extract context around the keyword
                context = self._extract_context(notes, keyword)
                if context:
                    growth_signals.append(context)
        
        # Concern indicators  
        concern_keywords = [
            'struggle', 'difficult', 'concern', 'issue', 'problem', 'lack',
            'weak', 'ineffective', 'inconsistent', 'unclear'
        ]
        
        for keyword in concern_keywords:
            if keyword in notes_lower:
                context = self._extract_context(notes, keyword)
                if context:
                    concern_signals.append(context)
        
        # Analyze score patterns for additional signals
        if component_scores:
            scores = [float(s) for s in component_scores.values() if isinstance(s, (int, float))]
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= 3.0:
                    growth_signals.append("Consistently high component scores")
                elif avg_score <= 1.5:
                    concern_signals.append("Below-average component performance")
        
        return growth_signals[:3], concern_signals[:3]  # Limit signals
    
    def _extract_context(self, text: str, keyword: str) -> Optional[str]:
        """Extract a sentence containing the keyword for context."""
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                sentence = sentence.strip()
                if len(sentence) > 10:  # Meaningful length
                    return sentence
        return None
    
    def _generate_domain_summary(self, domain_score) -> str:
        """Generate a brief summary for a domain."""
        if domain_score.notes:
            # Extract first meaningful sentence
            sentences = re.split(r'[.!?]+', domain_score.notes)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:
                    return sentence
        
        # Fallback based on score
        if domain_score.overall_score is None:
            return "No evaluation data available"
        
        score = float(domain_score.overall_score)
        if score >= 3.0:
            return "Demonstrates proficient or exemplary performance"
        elif score >= 2.0:
            return "Shows developing competency with room for growth"
        else:
            return "Requires focused support and development"
    
    async def _extract_evidence_snippets(
        self,
        evaluation: DanielsonEvaluation,
        max_snippets: int = 10
    ) -> List[str]:
        """Extract key evidence snippets from evaluation text."""
        evidence_snippets = []
        
        try:
            # Extract from low inference notes
            if evaluation.low_inference_notes:
                snippets = self._extract_quotes_from_text(evaluation.low_inference_notes)
                evidence_snippets.extend(snippets)
            
            # Extract from domain scores notes
            domain_scores = evaluation.get_domain_scores()
            for domain_score in domain_scores:
                if domain_score.notes:
                    snippets = self._extract_quotes_from_text(domain_score.notes)
                    evidence_snippets.extend(snippets)
            
            # Extract from main evaluation JSON if available
            if evaluation.evaluation:
                eval_text = self._extract_text_from_json(evaluation.evaluation)
                if eval_text:
                    snippets = self._extract_quotes_from_text(eval_text)
                    evidence_snippets.extend(snippets)
            
            # Remove duplicates and limit
            evidence_snippets = list(dict.fromkeys(evidence_snippets))  # Remove dupes
            evidence_snippets = evidence_snippets[:max_snippets]
            
        except Exception as e:
            self.logger.warning(f"Error extracting evidence: {e}")
        
        return evidence_snippets
    
    def _extract_text_from_json(self, json_data: Dict[str, Any]) -> str:
        """Extract text content from evaluation JSON structure."""
        text_parts = []
        
        def extract_text_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in ['notes', 'comment', 'description', 'evidence', 'observation']:
                        if isinstance(value, str) and len(value.strip()) > 10:
                            text_parts.append(value.strip())
                    else:
                        extract_text_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item)
        
        extract_text_recursive(json_data)
        return ' '.join(text_parts)
    
    async def _detect_risk_signals(self, evaluation: DanielsonEvaluation) -> RiskSignals:
        """Detect burnout, disengagement, and leaving risk signals."""
        risk_signals = RiskSignals()
        
        try:
            # Collect all text for analysis
            text_sources = []
            
            if evaluation.low_inference_notes:
                text_sources.append(evaluation.low_inference_notes)
            
            domain_scores = evaluation.get_domain_scores()
            for domain_score in domain_scores:
                if domain_score.notes:
                    text_sources.append(domain_score.notes)
            
            if evaluation.evaluation:
                eval_text = self._extract_text_from_json(evaluation.evaluation)
                if eval_text:
                    text_sources.append(eval_text)
            
            # Analyze text for risk keywords
            all_text = ' '.join(text_sources).lower()
            
            # Burnout detection
            for keyword in self.risk_keywords['burnout']:
                if keyword in all_text:
                    context = self._find_risk_context(all_text, keyword)
                    if context:
                        risk_signals.burnout_indicators.append(context)
            
            # Disengagement detection
            for keyword in self.risk_keywords['disengagement']:
                if keyword in all_text:
                    context = self._find_risk_context(all_text, keyword)
                    if context:
                        risk_signals.disengagement_signals.append(context)
            
            # Leaving risk detection
            for keyword in self.risk_keywords['leaving_risk']:
                if keyword in all_text:
                    context = self._find_risk_context(all_text, keyword)
                    if context:
                        risk_signals.leaving_risk_factors.append(context)
            
            # Calculate confidence score based on signal strength
            total_signals = (
                len(risk_signals.burnout_indicators) +
                len(risk_signals.disengagement_signals) +
                len(risk_signals.leaving_risk_factors)
            )
            
            # Simple confidence calculation
            if total_signals >= 3:
                risk_signals.confidence_score = 0.8
            elif total_signals >= 2:
                risk_signals.confidence_score = 0.6
            elif total_signals >= 1:
                risk_signals.confidence_score = 0.4
            else:
                risk_signals.confidence_score = 0.0
            
            # Limit signals to avoid noise
            risk_signals.burnout_indicators = risk_signals.burnout_indicators[:3]
            risk_signals.disengagement_signals = risk_signals.disengagement_signals[:3]
            risk_signals.leaving_risk_factors = risk_signals.leaving_risk_factors[:3]
            
        except Exception as e:
            self.logger.warning(f"Error detecting risk signals: {e}")
        
        return risk_signals
    
    def _find_risk_context(self, text: str, keyword: str) -> Optional[str]:
        """Find contextual sentence containing a risk keyword."""
        # Split into sentences around the keyword
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if keyword in sentence.lower():
                sentence = sentence.strip()
                if len(sentence) > 15:  # Meaningful context
                    return sentence[:100]  # Truncate for brevity
        return None
    
    def _generate_flags(
        self,
        domain_summaries: Dict[str, DomainSummary],
        risk_signals: RiskSignals,
        org_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate flags and indicators based on analysis."""
        flags = {
            "needs_PD": [],
            "exemplar": False,
            "risk_of_leaving": False,
            "burnout_signals": False
        }
        
        try:
            # Identify domains needing professional development (red/yellow)
            for domain_id, summary in domain_summaries.items():
                if summary.status_color in [DomainStatus.RED, DomainStatus.YELLOW]:
                    flags["needs_PD"].append(domain_id)
            
            # Check for exemplar performance
            green_count = sum(1 for s in domain_summaries.values() 
                            if s.status_color == DomainStatus.GREEN)
            total_evaluated = sum(1 for s in domain_summaries.values() 
                                if s.score is not None)
            
            if total_evaluated > 0:
                green_percentage = green_count / total_evaluated
                exemplar_threshold = org_config.get('global_thresholds', {}).get('exemplar_teacher', 0.8)
                
                # Convert score-based threshold to percentage-based check
                if green_percentage >= 0.8:  # 80% green domains
                    flags["exemplar"] = True
            
            # Risk flags
            if risk_signals.leaving_risk_factors:
                flags["risk_of_leaving"] = True
            
            if risk_signals.burnout_indicators:
                flags["burnout_signals"] = True
            
        except Exception as e:
            self.logger.warning(f"Error generating flags: {e}")
        
        return flags
    
    async def _extract_strengths_and_concerns(
        self,
        evaluation: DanielsonEvaluation,
        domain_summaries: Dict[str, DomainSummary]
    ) -> Tuple[List[str], List[str]]:
        """Extract key strengths and concerns from evaluation."""
        strengths = []
        concerns = []
        
        try:
            # Collect strengths from green domains
            for domain_id, summary in domain_summaries.items():
                if summary.status_color == DomainStatus.GREEN:
                    strengths.extend(summary.growth_signals)
                elif summary.status_color == DomainStatus.RED:
                    concerns.extend(summary.concern_signals)
            
            # Remove duplicates and limit
            strengths = list(dict.fromkeys(strengths))[:5]
            concerns = list(dict.fromkeys(concerns))[:5]
            
            # If we don't have enough, extract from evaluation text using LLM
            if len(strengths) < 2 or len(concerns) < 2:
                llm_strengths, llm_concerns = await self._llm_extract_strengths_concerns(evaluation)
                
                strengths.extend(llm_strengths)
                concerns.extend(llm_concerns)
                
                # Remove duplicates again and limit
                strengths = list(dict.fromkeys(strengths))[:5]
                concerns = list(dict.fromkeys(concerns))[:5]
            
        except Exception as e:
            self.logger.warning(f"Error extracting strengths/concerns: {e}")
        
        return strengths, concerns
    
    async def _llm_extract_strengths_concerns(
        self,
        evaluation: DanielsonEvaluation
    ) -> Tuple[List[str], List[str]]:
        """Use LLM to extract strengths and concerns from evaluation text."""
        try:
            # Prepare evaluation text
            text_sources = []
            
            if evaluation.low_inference_notes:
                text_sources.append(f"Low Inference Notes: {evaluation.low_inference_notes}")
            
            domain_scores = evaluation.get_domain_scores()
            for domain_score in domain_scores:
                if domain_score.notes:
                    text_sources.append(f"Domain {domain_score.domain_id}: {domain_score.notes}")
            
            eval_text = '\n\n'.join(text_sources)
            
            if not eval_text.strip():
                return [], []
            
            prompt = f"""
            Analyze the following teacher evaluation text and extract key strengths and areas of concern.
            
            Evaluation Text:
            {eval_text}
            
            Please provide:
            1. Top 3 key strengths (specific, evidence-based)
            2. Top 3 areas of concern (specific, actionable)
            
            Format as JSON:
            {{
                "strengths": ["strength1", "strength2", "strength3"],
                "concerns": ["concern1", "concern2", "concern3"]
            }}
            """
            
            class StrengthsConcerns(BaseModel):
                strengths: List[str] = []
                concerns: List[str] = []
            
            response = await self.llm_call(
                prompt=prompt,
                response_format=StrengthsConcerns,
                temperature=0.3
            )
            
            if isinstance(response, StrengthsConcerns):
                return response.strengths, response.concerns
            else:
                return [], []
                
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}")
            return [], []
    
    def _assess_relevance(self, evaluation: DanielsonEvaluation) -> str:
        """Assess how relevant this evaluation is to analysis questions."""
        # Simple heuristic based on data completeness
        
        has_notes = bool(evaluation.low_inference_notes and evaluation.low_inference_notes.strip())
        domain_scores = evaluation.get_domain_scores()
        has_domain_data = len(domain_scores) > 0
        
        if has_notes and has_domain_data:
            return "high"
        elif has_notes or has_domain_data:
            return "medium"
        else:
            return "low"