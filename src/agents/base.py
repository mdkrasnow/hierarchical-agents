"""Abstract base agent class with async LLM calls, structured output, and template support."""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

from utils.llm import LLMClient, LLMError, BatchResult
from agents.templates import TemplateManager, get_template_manager, render_prompt


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMetrics:
    """Metrics for agent execution."""
    total_llm_calls: int = 0
    total_llm_tokens: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def execution_time_ms(self) -> float:
        """Calculate total execution time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency per LLM call."""
        if self.total_llm_calls > 0:
            return self.total_latency_ms / self.total_llm_calls
        return 0.0


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_retry_count: int = 3
    max_concurrent_calls: int = 5
    template_loader: Optional[str] = None
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    enable_metrics: bool = True
    log_level: str = "INFO"


class AgentResult(BaseModel):
    """Standard result format for agent operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str
    execution_time_ms: float
    metrics: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base class for all hierarchical agents.
    
    Provides:
    - Async LLM calls with retry logic
    - Structured JSON output parsing  
    - Template-based prompt management
    - Batch processing capabilities
    - Comprehensive error handling and logging
    - Performance metrics collection
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        config: Optional[AgentConfig] = None,
        template_manager: Optional[TemplateManager] = None
    ):
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.llm_client = llm_client
        self.config = config or AgentConfig()
        self.template_manager = template_manager or get_template_manager()
        
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics() if self.config.enable_metrics else None
        self._execution_context: Dict[str, Any] = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the type/name of this agent."""
        pass
    
    @property
    @abstractmethod  
    def role_description(self) -> str:
        """Return a description of this agent's role."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Main execution method for the agent.
        
        Subclasses must implement this method to define their core functionality.
        """
        pass
    
    async def llm_call(
        self,
        prompt: str,
        response_format: Optional[Type[T]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[Any, T]:
        """
        Make a single LLM call with structured output parsing.
        
        Args:
            prompt: The prompt to send to the LLM
            response_format: Optional Pydantic model for structured output
            temperature: Override default temperature
            max_tokens: Override default max tokens
            metadata: Additional metadata for the call
            
        Returns:
            Parsed response (if response_format provided) or raw response
        """
        if not self.llm_client:
            raise LLMError("No LLM client configured")
        
        start_time = time.time()
        
        try:
            result = await self.llm_client.call(
                prompt=prompt,
                response_format=response_format,
                temperature=temperature or self.config.llm_temperature,
                max_tokens=max_tokens or self.config.llm_max_tokens,
                retry_count=self.config.llm_retry_count,
                metadata={
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    **(metadata or {})
                }
            )
            
            # Update metrics
            if self.metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.total_llm_calls += 1
                self.metrics.total_latency_ms += latency_ms
                
                # Extract token usage if available
                if hasattr(result, 'token_usage') and result.token_usage:
                    tokens = result.token_usage.get('input_tokens', 0) + result.token_usage.get('output_tokens', 0)
                    self.metrics.total_llm_tokens += tokens
            
            return result
            
        except Exception as e:
            if self.metrics:
                self.metrics.error_count += 1
            
            self.logger.error(f"LLM call failed: {e}", extra={
                "agent_id": self.agent_id,
                "prompt_length": len(prompt),
                "error_type": type(e).__name__
            })
            raise
    
    async def llm_call_batch(
        self,
        prompts: List[str],
        response_formats: Optional[List[Optional[Type[BaseModel]]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """
        Make multiple LLM calls in batch with concurrency control.
        
        Args:
            prompts: List of prompts to send
            response_formats: Optional list of Pydantic models for each prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            metadata: Additional metadata for all calls
            
        Returns:
            BatchResult with responses and failed requests
        """
        if not self.llm_client:
            raise LLMError("No LLM client configured")
        
        if response_formats and len(response_formats) != len(prompts):
            raise ValueError("response_formats length must match prompts length")
        
        response_formats = response_formats or [None] * len(prompts)
        requests = list(zip(prompts, response_formats))
        
        start_time = time.time()
        
        try:
            result = await self.llm_client.call_batch(
                requests=requests,
                temperature=temperature or self.config.llm_temperature,
                max_tokens=max_tokens or self.config.llm_max_tokens,
                metadata={
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "batch_size": len(prompts),
                    **(metadata or {})
                }
            )
            
            # Update metrics
            if self.metrics:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.total_llm_calls += len(prompts)
                self.metrics.total_latency_ms += latency_ms
                
                # Count tokens from successful responses
                for response in result.responses:
                    if response.token_usage:
                        tokens = response.token_usage.get('input_tokens', 0) + response.token_usage.get('output_tokens', 0)
                        self.metrics.total_llm_tokens += tokens
                
                self.metrics.error_count += result.failed_count
            
            self.logger.info(f"Batch LLM call completed: {result.successful_count}/{len(prompts)} successful")
            return result
            
        except Exception as e:
            if self.metrics:
                self.metrics.error_count += len(prompts)
            
            self.logger.error(f"Batch LLM call failed: {e}", extra={
                "agent_id": self.agent_id,
                "batch_size": len(prompts),
                "error_type": type(e).__name__
            })
            raise
    
    async def render_prompt(
        self,
        template_name: str,
        variables: Optional[Dict[str, Any]] = None,
        loader_name: Optional[str] = None
    ) -> str:
        """
        Render a prompt template with agent context.
        
        Args:
            template_name: Name of the template to render
            variables: Variables to substitute in the template
            loader_name: Optional specific loader to use
            
        Returns:
            Rendered prompt string
        """
        # Merge agent context with provided variables
        template_vars = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "role_description": self.role_description,
            **self._execution_context,
            **(variables or {})
        }
        
        try:
            return await self.template_manager.render_template(
                template_name=template_name,
                variables=template_vars,
                loader_name=loader_name or self.config.template_loader
            )
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}", extra={
                "agent_id": self.agent_id,
                "template_name": template_name,
                "variables": list(template_vars.keys())
            })
            raise
    
    async def llm_call_with_template(
        self,
        template_name: str,
        template_variables: Optional[Dict[str, Any]] = None,
        response_format: Optional[Type[T]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        loader_name: Optional[str] = None
    ) -> Union[Any, T]:
        """
        Convenience method to render a template and make an LLM call.
        
        Args:
            template_name: Name of the template to render
            template_variables: Variables for template rendering
            response_format: Optional Pydantic model for structured output
            temperature: Override default temperature
            max_tokens: Override default max tokens
            loader_name: Optional specific loader to use
            
        Returns:
            Parsed response (if response_format provided) or raw response
        """
        prompt = await self.render_prompt(template_name, template_variables, loader_name)
        return await self.llm_call(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata={"template_name": template_name}
        )
    
    def set_context(self, key: str, value: Any):
        """Set a value in the execution context."""
        self._execution_context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the execution context."""
        return self._execution_context.get(key, default)
    
    def clear_context(self):
        """Clear the execution context."""
        self._execution_context.clear()
    
    async def _start_execution(self):
        """Internal method to start execution tracking."""
        self.status = AgentStatus.RUNNING
        if self.metrics:
            self.metrics.start_time = time.time()
        
        self.logger.info(f"Starting execution for {self.agent_type} agent", extra={
            "agent_id": self.agent_id
        })
    
    async def _end_execution(self, success: bool, error: Optional[Exception] = None):
        """Internal method to end execution tracking."""
        self.status = AgentStatus.COMPLETED if success else AgentStatus.FAILED
        if self.metrics:
            self.metrics.end_time = time.time()
        
        log_level = logging.INFO if success else logging.ERROR
        message = f"Execution {'completed' if success else 'failed'} for {self.agent_type} agent"
        
        extra_data = {"agent_id": self.agent_id}
        if error:
            extra_data["error"] = str(error)
        if self.metrics:
            extra_data.update({
                "execution_time_ms": self.metrics.execution_time_ms,
                "llm_calls": self.metrics.total_llm_calls,
                "llm_tokens": self.metrics.total_llm_tokens,
                "errors": self.metrics.error_count
            })
        
        self.logger.log(log_level, message, extra=extra_data)
    
    async def execute_with_tracking(self, **kwargs) -> AgentResult:
        """
        Execute the agent with automatic status and metrics tracking.
        
        This method wraps the main execute() method with tracking logic.
        """
        await self._start_execution()
        
        try:
            result = await self.execute(**kwargs)
            await self._end_execution(success=result.success)
            return result
            
        except Exception as e:
            await self._end_execution(success=False, error=e)
            
            # Return error result instead of re-raising
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                execution_time_ms=self.metrics.execution_time_ms if self.metrics else 0.0,
                metadata={
                    "error_type": type(e).__name__,
                    "agent_type": self.agent_type
                },
                metrics=self._get_metrics_dict() if self.metrics else None
            )
    
    def _get_metrics_dict(self) -> Optional[Dict[str, Any]]:
        """Get metrics as a dictionary."""
        if not self.metrics:
            return None
        
        return {
            "total_llm_calls": self.metrics.total_llm_calls,
            "total_llm_tokens": self.metrics.total_llm_tokens,
            "total_latency_ms": self.metrics.total_latency_ms,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "error_count": self.metrics.error_count,
            "execution_time_ms": self.metrics.execution_time_ms
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, status={self.status.value})"