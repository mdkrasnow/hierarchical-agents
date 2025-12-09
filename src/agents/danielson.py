"""
Danielson-specific evaluation agent implementation.

This module provides Danielson framework-specific processing logic,
including domain interpretation, rubric application, and framework-specific
risk detection patterns.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.evaluation import DanielsonEvaluationAgent, EvaluationInput, RiskSignals
from models import DomainSummary, DomainStatus, DanielsonEvaluation
from utils.llm import LLMClient


logger = logging.getLogger(__name__)


class DanielsonSpecificAgent(DanielsonEvaluationAgent):
    """
    Danielson framework-specific evaluation agent.
    
    Extends the base evaluation agent with Danielson-specific:
    - Domain interpretation and rubric application
    - Framework-specific evidence patterns
    - Professional responsibility indicators
    - Teaching practice analysis
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        **kwargs
    ):
        super().__init__(llm_client=llm_client, **kwargs)
        
        # Danielson-specific domain groupings
        self.domain_groups = {
            'planning': ['I-A', 'I-B', 'I-C', 'I-D', 'I-E', 'I-F'],
            'environment': ['II-A', 'II-B', 'II-C', 'II-D', 'II-E'],
            'instruction': ['III-A', 'III-B', 'III-C', 'III-D', 'III-E'],
            'professional': ['IV-A', 'IV-B', 'IV-C', 'IV-D', 'IV-E', 'IV-F']
        }
        
        # Domain importance weights for analysis
        self.domain_priorities = {
            # Core instruction domains (highest priority)
            'III-B': 1.0,  # Questioning and Discussion
            'III-C': 1.0,  # Engaging Students in Learning  
            'II-B': 0.9,   # Culture for Learning
            'III-A': 0.9,  # Communicating with Students
            
            # Important foundation domains
            'I-A': 0.8,    # Knowledge of Content and Pedagogy
            'I-C': 0.8,    # Setting Instructional Outcomes
            'II-A': 0.8,   # Environment of Respect and Rapport
            
            # Supporting domains  
            'I-B': 0.7,    # Knowledge of Students
            'III-D': 0.7,  # Using Assessment
            'IV-A': 0.7,   # Reflecting on Teaching
            
            # Additional domains (standard weight)
            'I-D': 0.6, 'I-E': 0.6, 'I-F': 0.6,
            'II-C': 0.6, 'II-D': 0.6, 'II-E': 0.6,  
            'III-E': 0.6,
            'IV-B': 0.6, 'IV-C': 0.6, 'IV-D': 0.6, 'IV-E': 0.6, 'IV-F': 0.6
        }
        
        # Danielson-specific evidence patterns
        self.evidence_patterns = {
            'strong_instruction': [
                'clear learning objectives', 'student engagement', 'differentiated instruction',
                'effective questioning', 'feedback to students', 'assessment for learning'
            ],
            'classroom_management': [
                'positive classroom culture', 'respectful interactions', 'established routines',
                'student behavior management', 'learning environment'
            ],
            'professional_growth': [
                'reflection on practice', 'professional development', 'collaboration with colleagues',
                'communication with families', 'advocacy for students'
            ]
        }
    
    @property
    def agent_type(self) -> str:
        return "DanielsonSpecificAgent"
    
    @property  
    def role_description(self) -> str:
        return "Process Danielson framework evaluations with domain-specific analysis and rubric interpretation"
    
    async def _process_domain_scores(
        self,
        evaluation: DanielsonEvaluation,
        org_config: Dict[str, Any]
    ) -> Dict[str, DomainSummary]:
        """
        Enhanced domain processing with Danielson-specific logic.
        """
        # Get base domain summaries
        domain_summaries = await super()._process_domain_scores(evaluation, org_config)
        
        # Apply Danielson-specific enhancements
        await self._enhance_danielson_domains(domain_summaries, evaluation, org_config)
        
        return domain_summaries
    
    async def _enhance_danielson_domains(
        self,
        domain_summaries: Dict[str, DomainSummary],
        evaluation: DanielsonEvaluation,
        org_config: Dict[str, Any]
    ):
        """Apply Danielson-specific enhancements to domain summaries."""
        
        for domain_id, summary in domain_summaries.items():
            # Apply domain-specific analysis
            if domain_id in ['III-B', 'III-C']:  # Core instruction domains
                await self._analyze_instruction_domain(summary, evaluation, domain_id)
            elif domain_id in ['II-A', 'II-B']:  # Environment domains  
                await self._analyze_environment_domain(summary, evaluation, domain_id)
            elif domain_id in ['I-A', 'I-C']:  # Planning domains
                await self._analyze_planning_domain(summary, evaluation, domain_id)
            elif domain_id.startswith('IV'):  # Professional domains
                await self._analyze_professional_domain(summary, evaluation, domain_id)
            
            # Store priority weighting in summary metadata (not as a field)
            if domain_id in self.domain_priorities:
                # Could add to summary text or store separately for later use
                pass
    
    async def _analyze_instruction_domain(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation,
        domain_id: str
    ):
        """Analyze core instruction domains (III-B, III-C) with specific patterns."""
        
        if domain_id == 'III-B':  # Questioning and Discussion
            # Look for questioning evidence
            questioning_evidence = self._find_questioning_evidence(summary, evaluation)
            if questioning_evidence:
                summary.evidence_quotes.extend(questioning_evidence)
                summary.growth_signals.append("Evidence of effective questioning strategies")
            else:
                summary.concern_signals.append("Limited evidence of questioning techniques")
        
        elif domain_id == 'III-C':  # Engaging Students
            # Look for engagement evidence  
            engagement_evidence = self._find_engagement_evidence(summary, evaluation)
            if engagement_evidence:
                summary.evidence_quotes.extend(engagement_evidence)
                summary.growth_signals.append("Clear student engagement strategies")
            else:
                summary.concern_signals.append("Student engagement needs development")
    
    async def _analyze_environment_domain(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation,
        domain_id: str
    ):
        """Analyze classroom environment domains."""
        
        # Look for environment-related evidence
        env_keywords = [
            'respectful', 'rapport', 'culture', 'learning environment',
            'classroom management', 'positive interactions'
        ]
        
        domain_text = self._get_domain_text(summary, evaluation, domain_id)
        env_signals = self._extract_signals_by_keywords(domain_text, env_keywords)
        
        if env_signals:
            summary.growth_signals.extend(env_signals['positive'])
            summary.concern_signals.extend(env_signals['negative'])
    
    async def _analyze_planning_domain(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation,
        domain_id: str
    ):
        """Analyze planning and preparation domains."""
        
        planning_keywords = [
            'lesson plans', 'learning objectives', 'instructional outcomes',
            'content knowledge', 'pedagogy', 'student needs'
        ]
        
        domain_text = self._get_domain_text(summary, evaluation, domain_id)
        planning_signals = self._extract_signals_by_keywords(domain_text, planning_keywords)
        
        if planning_signals:
            summary.growth_signals.extend(planning_signals['positive'])
            summary.concern_signals.extend(planning_signals['negative'])
    
    async def _analyze_professional_domain(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation,
        domain_id: str
    ):
        """Analyze professional responsibilities domains."""
        
        professional_keywords = [
            'reflection', 'professional development', 'collaboration',
            'communication', 'advocacy', 'growth', 'improvement'
        ]
        
        domain_text = self._get_domain_text(summary, evaluation, domain_id)
        prof_signals = self._extract_signals_by_keywords(domain_text, professional_keywords)
        
        if prof_signals:
            summary.growth_signals.extend(prof_signals['positive'])
            summary.concern_signals.extend(prof_signals['negative'])
    
    def _find_questioning_evidence(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation
    ) -> List[str]:
        """Find evidence related to questioning and discussion techniques."""
        
        questioning_patterns = [
            r'question(?:ing|s?)',
            r'discussion',
            r'student.*respond',
            r'higher.*order.*thinking',
            r'critical.*thinking',
            r'wait.*time'
        ]
        
        domain_text = self._get_domain_text(summary, evaluation, 'III-B')
        evidence = self._extract_pattern_matches(domain_text, questioning_patterns)
        
        return evidence[:3]  # Limit to top 3 pieces of evidence
    
    def _find_engagement_evidence(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation
    ) -> List[str]:
        """Find evidence related to student engagement."""
        
        engagement_patterns = [
            r'engag(?:ed|ing|ement)',
            r'student.*participation',
            r'active.*learning',
            r'motivated',
            r'hands.*on',
            r'collaborative'
        ]
        
        domain_text = self._get_domain_text(summary, evaluation, 'III-C')
        evidence = self._extract_pattern_matches(domain_text, engagement_patterns)
        
        return evidence[:3]
    
    def _get_domain_text(
        self,
        summary: DomainSummary,
        evaluation: DanielsonEvaluation,
        domain_id: str
    ) -> str:
        """Get all text related to a specific domain."""
        
        text_parts = []
        
        # Get text from domain summary
        if summary.summary:
            text_parts.append(summary.summary)
        
        # Get text from evidence quotes
        text_parts.extend(summary.evidence_quotes)
        
        # Get text from evaluation domain scores
        domain_scores = evaluation.get_domain_scores()
        for domain_score in domain_scores:
            if domain_score.domain_id == domain_id and domain_score.notes:
                text_parts.append(domain_score.notes)
        
        return ' '.join(text_parts)
    
    def _extract_signals_by_keywords(
        self,
        text: str,
        keywords: List[str]
    ) -> Dict[str, List[str]]:
        """Extract positive and negative signals based on keywords."""
        
        if not text:
            return {'positive': [], 'negative': []}
        
        text_lower = text.lower()
        positive_signals = []
        negative_signals = []
        
        # Positive indicators
        positive_modifiers = ['strong', 'effective', 'excellent', 'clear', 'well', 'good']
        negative_modifiers = ['weak', 'unclear', 'poor', 'lacking', 'insufficient', 'needs']
        
        for keyword in keywords:
            if keyword in text_lower:
                context = self._extract_context(text, keyword)
                if context:
                    context_lower = context.lower()
                    
                    # Check for positive or negative context
                    is_positive = any(mod in context_lower for mod in positive_modifiers)
                    is_negative = any(mod in context_lower for mod in negative_modifiers)
                    
                    if is_positive and not is_negative:
                        positive_signals.append(context)
                    elif is_negative and not is_positive:
                        negative_signals.append(context)
        
        return {
            'positive': positive_signals[:3],
            'negative': negative_signals[:3]
        }
    
    def _extract_pattern_matches(
        self,
        text: str,
        patterns: List[str]
    ) -> List[str]:
        """Extract text matching specific regex patterns."""
        
        import re
        matches = []
        
        if not text:
            return matches
        
        for pattern in patterns:
            regex_matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in regex_matches:
                # Extract sentence containing the match
                start = max(0, text.rfind('.', 0, match.start()) + 1)
                end = text.find('.', match.end())
                if end == -1:
                    end = len(text)
                
                sentence = text[start:end].strip()
                if len(sentence) > 15 and sentence not in matches:
                    matches.append(sentence)
                    
                if len(matches) >= 5:  # Limit matches
                    break
        
        return matches
    
    async def _detect_risk_signals(self, evaluation: DanielsonEvaluation) -> RiskSignals:
        """Enhanced risk detection with Danielson-specific patterns."""
        
        # Get base risk signals
        risk_signals = await super()._detect_risk_signals(evaluation)
        
        # Add Danielson-specific risk detection
        await self._detect_danielson_risks(risk_signals, evaluation)
        
        return risk_signals
    
    async def _detect_danielson_risks(
        self,
        risk_signals: RiskSignals,
        evaluation: DanielsonEvaluation
    ):
        """Detect risks specific to Danielson framework patterns."""
        
        # Analyze domain pattern for systemic issues
        domain_scores = evaluation.get_domain_scores()
        
        # Check for concerning domain patterns
        planning_scores = []
        instruction_scores = []
        professional_scores = []
        
        for domain_score in domain_scores:
            domain_id = domain_score.domain_id
            if domain_score.overall_score is not None:
                score = float(domain_score.overall_score)
                
                if domain_id.startswith('I'):  # Planning
                    planning_scores.append(score)
                elif domain_id.startswith('III'):  # Instruction  
                    instruction_scores.append(score)
                elif domain_id.startswith('IV'):  # Professional
                    professional_scores.append(score)
        
        # Check for systemic weaknesses
        if instruction_scores and sum(instruction_scores) / len(instruction_scores) < 2.0:
            risk_signals.disengagement_signals.append(
                "Consistently low instructional performance scores"
            )
        
        if professional_scores and sum(professional_scores) / len(professional_scores) < 2.0:
            risk_signals.burnout_indicators.append(
                "Declining professional responsibilities engagement" 
            )
        
        # Look for specific Danielson risk patterns in text
        all_text = self._get_all_evaluation_text(evaluation).lower()
        
        danielson_risk_patterns = {
            'burnout': [
                'overwhelmed by', 'too much', 'can\'t keep up',
                'exhausted', 'stressed about', 'workload'
            ],
            'disengagement': [
                'going through motions', 'minimal', 'just getting by',
                'not enthusiastic', 'checking boxes'
            ],
            'leaving': [
                'thinking about', 'considering other', 'maybe time',
                'tired of teaching', 'looking for'
            ]
        }
        
        for pattern in danielson_risk_patterns['burnout']:
            if pattern in all_text:
                context = self._find_risk_context(all_text, pattern)
                if context and context not in risk_signals.burnout_indicators:
                    risk_signals.burnout_indicators.append(context)
        
        for pattern in danielson_risk_patterns['disengagement']:
            if pattern in all_text:
                context = self._find_risk_context(all_text, pattern)
                if context and context not in risk_signals.disengagement_signals:
                    risk_signals.disengagement_signals.append(context)
        
        for pattern in danielson_risk_patterns['leaving']:
            if pattern in all_text:
                context = self._find_risk_context(all_text, pattern)
                if context and context not in risk_signals.leaving_risk_factors:
                    risk_signals.leaving_risk_factors.append(context)
        
        # Limit signals to prevent noise
        risk_signals.burnout_indicators = risk_signals.burnout_indicators[:5]
        risk_signals.disengagement_signals = risk_signals.disengagement_signals[:5]
        risk_signals.leaving_risk_factors = risk_signals.leaving_risk_factors[:5]
    
    def _get_all_evaluation_text(self, evaluation: DanielsonEvaluation) -> str:
        """Get all text content from an evaluation."""
        
        text_parts = []
        
        if evaluation.low_inference_notes:
            text_parts.append(evaluation.low_inference_notes)
        
        domain_scores = evaluation.get_domain_scores()
        for domain_score in domain_scores:
            if domain_score.notes:
                text_parts.append(domain_score.notes)
        
        if evaluation.evaluation:
            eval_text = self._extract_text_from_json(evaluation.evaluation)
            if eval_text:
                text_parts.append(eval_text)
        
        return ' '.join(text_parts)
    
    async def calculate_danielson_priority_score(
        self,
        domain_summaries: Dict[str, DomainSummary]
    ) -> float:
        """
        Calculate a priority score based on Danielson domain weights.
        
        This can be used by higher-level agents to prioritize teachers
        based on performance in critical domains.
        """
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for domain_id, summary in domain_summaries.items():
            if summary.score is not None and domain_id in self.domain_priorities:
                weight = self.domain_priorities[domain_id]
                score = float(summary.score)
                
                total_weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0
    
    async def get_danielson_focus_domains(
        self,
        domain_summaries: Dict[str, DomainSummary],
        max_domains: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Get prioritized list of domains needing focus for PD.
        
        Returns list of (domain_id, reason) tuples.
        """
        
        focus_domains = []
        
        # Priority order for focusing (most impactful domains first)
        priority_order = [
            'III-B', 'III-C', 'II-B', 'III-A', 'I-A', 'I-C', 'II-A'
        ]
        
        for domain_id in priority_order:
            if domain_id in domain_summaries:
                summary = domain_summaries[domain_id]
                
                # Focus on red domains first, then yellow
                if summary.status_color == DomainStatus.RED:
                    reason = f"Critical need: {summary.summary}"
                    focus_domains.append((domain_id, reason))
                elif summary.status_color == DomainStatus.YELLOW and len(focus_domains) < max_domains:
                    reason = f"Development opportunity: {summary.summary}"  
                    focus_domains.append((domain_id, reason))
            
            if len(focus_domains) >= max_domains:
                break
        
        return focus_domains