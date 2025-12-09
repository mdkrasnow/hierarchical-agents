"""
LLM-based answer generation using generic language models.

Provides direct LLM-based answer generation with support for different
prompting strategies including chain-of-thought reasoning.
"""

import time
from typing import Any, Dict, List, Optional

from utils.llm import LLMClient, LLMError
from agents.templates import get_template_manager

from .answer_generator import (
    AnswerGenerator,
    GenerationConfig,
    GenerationResult,
    GenerationStrategy
)


class LLMGenerator(AnswerGenerator):
    """
    Answer generator using direct LLM calls with configurable prompting strategies.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize LLM generator.
        
        Args:
            llm_client: LLM client for API calls
            config: Generation configuration
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.template_manager = get_template_manager()
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this generation strategy."""
        if self.config.use_chain_of_thought:
            return "llm_chain_of_thought"
        return "llm_generic"
    
    @property
    def supported_question_types(self) -> List[str]:
        """Return list of question types this generator supports well."""
        return [
            "general-writing",
            "analysis",
            "explanation", 
            "creative",
            "factual",
            "opinion",
            "comparison",
            "summarization"
        ]
    
    async def generate_answer(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        config_override: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate an answer using LLM."""
        start_time = time.time()
        
        if not self.llm_client:
            return self._create_base_result(
                success=False,
                start_time=start_time
            ).model_copy(update={
                "error_message": "No LLM client configured"
            })
        
        # Merge configuration
        effective_config = self._merge_configs(config_override)
        context = context or {}
        
        # Create initial result
        result = self._create_base_result(start_time=start_time)
        result.add_reasoning_step("Starting LLM-based answer generation")
        
        try:
            # Choose prompting strategy
            if effective_config.use_chain_of_thought:
                answer = await self._generate_with_cot(
                    question, context, effective_config, result
                )
            else:
                answer = await self._generate_direct(
                    question, context, effective_config, result
                )
            
            # Validate answer if enabled
            if effective_config.enable_validation:
                validation = await self.validate_answer(question, answer, context)
                if not validation["is_valid"]:
                    if effective_config.retry_on_failure:
                        result.add_warning("Initial answer failed validation, retrying")
                        answer = await self._retry_generation(
                            question, context, effective_config, result, validation
                        )
                    else:
                        result.add_warning("Generated answer failed validation")
                
                result.confidence_score = validation.get("quality_score", 0.5)
                result.estimated_quality = self._estimate_quality_tier(validation.get("quality_score", 0.5))
            
            # Finalize result
            result.answer = answer
            result.generation_time_ms = (time.time() - start_time) * 1000
            result.add_reasoning_step(f"Generation completed successfully with {len(answer.split())} words")
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            result.set_error(str(e))
            result.generation_time_ms = (time.time() - start_time) * 1000
            return result
    
    async def _generate_direct(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult
    ) -> str:
        """Generate answer using direct prompting."""
        result.add_reasoning_step("Using direct prompting strategy")
        
        # Build prompt
        prompt = await self._build_answer_prompt(question, context, config, cot=False)
        result.intermediate_outputs["prompt"] = prompt
        
        # Make LLM call
        response = await self.llm_client.call(
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract answer from response
        if hasattr(response, 'content'):
            answer = response.content.strip()
        else:
            answer = str(response).strip()
        
        result.llm_calls_made += 1
        if hasattr(response, 'token_usage') and response.token_usage:
            result.total_tokens_used += response.token_usage.get('input_tokens', 0)
            result.total_tokens_used += response.token_usage.get('output_tokens', 0)
        
        return answer
    
    async def _generate_with_cot(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult
    ) -> str:
        """Generate answer using chain-of-thought reasoning."""
        result.add_reasoning_step("Using chain-of-thought prompting strategy")
        
        # Build CoT prompt
        prompt = await self._build_answer_prompt(question, context, config, cot=True)
        result.intermediate_outputs["cot_prompt"] = prompt
        
        # Make LLM call for reasoning
        response = await self.llm_client.call(
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract reasoning and answer
        if hasattr(response, 'content'):
            full_response = response.content.strip()
        else:
            full_response = str(response).strip()
        
        # Parse reasoning and final answer
        reasoning, answer = self._parse_cot_response(full_response)
        
        result.reasoning_trace.extend(reasoning)
        result.intermediate_outputs["cot_reasoning"] = reasoning
        result.intermediate_outputs["full_cot_response"] = full_response
        
        result.llm_calls_made += 1
        if hasattr(response, 'token_usage') and response.token_usage:
            result.total_tokens_used += response.token_usage.get('input_tokens', 0)
            result.total_tokens_used += response.token_usage.get('output_tokens', 0)
        
        return answer
    
    async def _retry_generation(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult,
        validation: Dict[str, Any]
    ) -> str:
        """Retry generation with improved prompting."""
        result.add_reasoning_step("Retrying generation with validation feedback")
        
        # Add validation feedback to context
        retry_context = context.copy()
        retry_context["validation_issues"] = validation.get("issues", [])
        retry_context["quality_recommendations"] = validation.get("recommendations", [])
        
        # Try up to max_retries times
        for attempt in range(config.max_retries):
            try:
                # Generate with feedback context
                if config.use_chain_of_thought:
                    answer = await self._generate_with_cot(question, retry_context, config, result)
                else:
                    answer = await self._generate_direct(question, retry_context, config, result)
                
                # Re-validate
                new_validation = await self.validate_answer(question, answer, context)
                if new_validation["is_valid"]:
                    result.add_reasoning_step(f"Retry {attempt + 1} successful")
                    return answer
                else:
                    result.add_warning(f"Retry {attempt + 1} still failed validation")
                    
            except Exception as e:
                result.add_warning(f"Retry {attempt + 1} failed with error: {e}")
        
        # If all retries failed, return best attempt
        result.add_warning("All retry attempts failed, returning last attempt")
        return answer
    
    async def _build_answer_prompt(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        cot: bool = False
    ) -> str:
        """Build the prompt for answer generation."""
        
        # Choose template based on strategy
        template_name = "answer_generation_cot" if cot else "answer_generation_direct"
        
        # Prepare template variables
        variables = {
            "question": question,
            "custom_instructions": config.custom_instructions or "",
            "target_length": config.target_length,
            "include_reasoning": config.include_reasoning,
            **context,
            **config.generation_context
        }
        
        try:
            # Try to use template if available
            prompt = await self.template_manager.render_template(
                template_name=template_name,
                variables=variables
            )
        except Exception:
            # Fallback to default prompts if template not found
            prompt = self._build_fallback_prompt(question, context, config, cot)
        
        return prompt
    
    def _build_fallback_prompt(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        cot: bool = False
    ) -> str:
        """Build a fallback prompt when templates are not available."""
        
        prompt_parts = []
        
        if cot:
            prompt_parts.append(
                "Please answer the following question. Think step by step and show your reasoning before giving your final answer."
            )
        else:
            prompt_parts.append(
                "Please answer the following question clearly and completely."
            )
        
        # Add custom instructions if provided
        if config.custom_instructions:
            prompt_parts.append(f"Additional instructions: {config.custom_instructions}")
        
        # Add target length if specified
        if config.target_length:
            prompt_parts.append(f"Target answer length: approximately {config.target_length} words.")
        
        # Add context information
        if context:
            context_items = [f"{k}: {v}" for k, v in context.items() if v is not None]
            if context_items:
                prompt_parts.append(f"Context: {'; '.join(context_items)}")
        
        # Add the question
        prompt_parts.append(f"Question: {question}")
        
        if cot:
            prompt_parts.append(
                "Please think through this step by step, then provide your final answer."
                "Format your response as:\nReasoning: [your step-by-step thinking]\nFinal Answer: [your complete answer]"
            )
        else:
            prompt_parts.append("Answer:")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_cot_response(self, response: str) -> tuple[List[str], str]:
        """Parse chain-of-thought response to extract reasoning and answer."""
        reasoning_steps = []
        answer = response
        
        # Try to parse structured CoT response
        if "Reasoning:" in response and "Final Answer:" in response:
            parts = response.split("Final Answer:", 1)
            if len(parts) == 2:
                reasoning_part = parts[0].replace("Reasoning:", "").strip()
                answer = parts[1].strip()
                
                # Split reasoning into steps
                reasoning_steps = [
                    step.strip() for step in reasoning_part.split('\n') 
                    if step.strip() and not step.strip().startswith('Final Answer:')
                ]
        else:
            # Try to extract reasoning from step-by-step format
            lines = response.split('\n')
            reasoning_lines = []
            answer_lines = []
            in_answer = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for answer indicators
                if any(indicator in line.lower() for indicator in ['final answer', 'answer:', 'conclusion:']):
                    in_answer = True
                    # Include this line in answer if it has content after the indicator
                    colon_idx = line.find(':')
                    if colon_idx != -1 and len(line) > colon_idx + 1:
                        answer_lines.append(line[colon_idx + 1:].strip())
                    continue
                
                if in_answer:
                    answer_lines.append(line)
                else:
                    reasoning_lines.append(line)
            
            if answer_lines:
                answer = ' '.join(answer_lines)
                reasoning_steps = reasoning_lines
            else:
                # If no clear answer section found, use last part as answer
                if reasoning_lines:
                    answer = reasoning_lines[-1]
                    reasoning_steps = reasoning_lines[:-1]
        
        return reasoning_steps, answer
    
    def _estimate_quality_tier(self, quality_score: float) -> str:
        """Estimate quality tier based on validation score."""
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "fair"
        else:
            return "poor"