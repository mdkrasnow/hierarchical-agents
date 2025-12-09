"""Example agent implementations demonstrating the BaseAgent framework."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from .base import BaseAgent, AgentResult


class AnalysisResult(BaseModel):
    """Structured output for analysis tasks."""
    findings: list[str] = Field(description="Key findings from the analysis")
    recommendations: list[str] = Field(description="Recommended actions")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level in the analysis")
    next_steps: Optional[list[str]] = Field(default=None, description="Suggested next steps")


class SimpleAnalyzerAgent(BaseAgent):
    """Example agent that performs basic text analysis using LLM calls."""
    
    @property
    def agent_type(self) -> str:
        return "Text Analyzer"
    
    @property
    def role_description(self) -> str:
        return "Analyze text content and provide structured insights and recommendations"
    
    async def execute(self, text: str, analysis_type: str = "general", **kwargs) -> AgentResult:
        """
        Analyze the provided text and return structured insights.
        
        Args:
            text: The text content to analyze
            analysis_type: Type of analysis to perform (general, sentiment, technical, etc.)
            
        Returns:
            AgentResult with structured analysis findings
        """
        try:
            # Set context for this execution
            self.set_context("text_length", len(text))
            self.set_context("analysis_type", analysis_type)
            
            # Render prompt using template
            try:
                prompt = await self.render_prompt(
                    "base_agent",
                    {
                        "responsibilities": f"• Analyze {analysis_type} aspects of provided text\n• Identify key insights and patterns\n• Provide actionable recommendations",
                        "context": f"Analyzing {len(text)} characters of text for {analysis_type} insights",
                        "additional_instructions": f"Focus specifically on {analysis_type} analysis. Provide confidence scores for your findings."
                    },
                    loader_name="file"
                )
            except Exception as template_error:
                # Fallback to a simple prompt if template fails
                self.logger.warning(f"Template rendering failed, using fallback: {template_error}")
                prompt = f"""You are a {self.agent_type}.

Your role: {self.role_description}

Please analyze the following text for {analysis_type} insights and provide structured output:"""
            
            # Add the actual analysis task
            analysis_prompt = f"""{prompt}

## Text to Analyze
{text}

## Analysis Task
Please perform a {analysis_type} analysis of the above text. Provide your response in the following JSON format:

{{
    "findings": ["finding 1", "finding 2", "..."],
    "recommendations": ["recommendation 1", "recommendation 2", "..."],
    "confidence": 0.85,
    "next_steps": ["step 1", "step 2", "..."]
}}

Focus on providing specific, actionable insights."""
            
            # Make LLM call with structured output
            analysis_result = await self.llm_call(
                prompt=analysis_prompt,
                response_format=AnalysisResult,
                metadata={"analysis_type": analysis_type, "text_length": len(text)}
            )
            
            return AgentResult(
                success=True,
                data={
                    "analysis": analysis_result.model_dump() if hasattr(analysis_result, 'model_dump') else analysis_result,
                    "text_length": len(text),
                    "analysis_type": analysis_type
                },
                agent_id=self.agent_id,
                execution_time_ms=self.metrics.execution_time_ms if self.metrics else 0.0,
                metrics=self._get_metrics_dict()
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=self.metrics.execution_time_ms if self.metrics else 0.0,
                metadata={"error_type": type(e).__name__}
            )


class BatchProcessingAgent(BaseAgent):
    """Example agent demonstrating batch processing capabilities."""
    
    @property
    def agent_type(self) -> str:
        return "Batch Processor"
    
    @property
    def role_description(self) -> str:
        return "Process multiple items in parallel using batch LLM operations"
    
    async def execute(self, items: list[str], processing_type: str = "classify", **kwargs) -> AgentResult:
        """
        Process multiple items in batch.
        
        Args:
            items: List of items to process
            processing_type: Type of processing to perform
            
        Returns:
            AgentResult with batch processing results
        """
        try:
            if not items:
                return AgentResult(
                    success=True,
                    data={"results": [], "processed_count": 0},
                    agent_id=self.agent_id,
                    execution_time_ms=0.0
                )
            
            # Create prompts for each item
            prompts = []
            for i, item in enumerate(items):
                prompt = f"""You are a {processing_type} specialist.

Please {processing_type} the following item and provide a brief analysis:

Item #{i+1}: {item}

Provide your response in this format:
- Category: [your classification]
- Reasoning: [brief explanation]
- Confidence: [0.0-1.0]"""
                prompts.append(prompt)
            
            # Process in batch
            batch_result = await self.llm_call_batch(
                prompts=prompts,
                metadata={"processing_type": processing_type, "item_count": len(items)}
            )
            
            # Compile results
            results = []
            for i, response in enumerate(batch_result.responses):
                results.append({
                    "item_index": i,
                    "item": items[i],
                    "result": response.content,
                    "latency_ms": response.latency_ms
                })
            
            # Handle failed requests
            failed_items = []
            for item_index, error in batch_result.failed_requests:
                failed_items.append({
                    "item_index": item_index,
                    "item": items[item_index],
                    "error": str(error)
                })
            
            return AgentResult(
                success=True,
                data={
                    "results": results,
                    "failed_items": failed_items,
                    "processed_count": len(results),
                    "failed_count": len(failed_items),
                    "total_latency_ms": batch_result.total_latency_ms
                },
                agent_id=self.agent_id,
                execution_time_ms=self.metrics.execution_time_ms if self.metrics else 0.0,
                metrics=self._get_metrics_dict()
            )
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=self.metrics.execution_time_ms if self.metrics else 0.0,
                metadata={"error_type": type(e).__name__}
            )