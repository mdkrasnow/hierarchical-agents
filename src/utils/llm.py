"""LLM client utilities with async support, retry logic, and batch processing."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from contextlib import asynccontextmanager

import aiohttp
from pydantic import BaseModel, ValidationError as PydanticValidationError


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Exception raised when rate limit is exceeded."""
    pass


class ValidationError(LLMError):
    """Exception raised when LLM response doesn't match expected schema."""
    pass


class APIError(LLMError):
    """Exception raised for API-specific errors."""
    pass


@dataclass
class LLMRequest:
    """Represents a single LLM request."""
    prompt: str
    response_format: Optional[Type[BaseModel]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class LLMResponse:
    """Represents a single LLM response."""
    content: str
    parsed_data: Optional[BaseModel] = None
    metadata: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None


@dataclass
class BatchResult:
    """Results from a batch LLM operation."""
    responses: List[LLMResponse]
    failed_requests: List[tuple[int, Exception]]  # (index, error)
    total_latency_ms: float
    successful_count: int
    failed_count: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def call_single(self, request: LLMRequest) -> LLMResponse:
        """Make a single LLM call."""
        pass
    
    @abstractmethod
    async def call_batch(self, requests: List[LLMRequest]) -> BatchResult:
        """Make multiple LLM calls in batch."""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, delay_ms: float = 100, fail_rate: float = 0.0):
        self.delay_ms = delay_ms
        self.fail_rate = fail_rate
        self._call_count = 0
    
    async def call_single(self, request: LLMRequest) -> LLMResponse:
        """Mock implementation that returns structured responses."""
        self._call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)
        
        if self.fail_rate > 0 and (self._call_count % int(1 / self.fail_rate)) == 0:
            raise APIError("Mock API error")
        
        # Generate mock response based on response format
        if request.response_format:
            try:
                # Create a mock instance with default values
                mock_data = request.response_format()
                content = mock_data.model_dump_json()
                parsed_data = mock_data
            except Exception:
                content = '{"mock": "response"}'
                parsed_data = None
        else:
            content = "Mock LLM response"
            parsed_data = None
        
        return LLMResponse(
            content=content,
            parsed_data=parsed_data,
            latency_ms=self.delay_ms,
            token_usage={"input_tokens": 50, "output_tokens": 25}
        )
    
    async def call_batch(self, requests: List[LLMRequest]) -> BatchResult:
        """Process requests in parallel with mock provider."""
        start_time = time.time()
        
        tasks = [self.call_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        failed_requests = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_requests.append((i, result))
            else:
                responses.append(result)
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return BatchResult(
            responses=responses,
            failed_requests=failed_requests,
            total_latency_ms=total_latency_ms,
            successful_count=len(responses),
            failed_count=len(failed_requests)
        )


class ClaudeLLMProvider(LLMProvider):
    """Claude API provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        base_url: str = "https://api.anthropic.com/v1/messages",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
    async def call_single(self, request: LLMRequest) -> LLMResponse:
        """Make a single Claude API call with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "messages": [
                {"role": "user", "content": request.prompt}
            ]
        }
        
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 429:
                            raise RateLimitError("Rate limit exceeded")
                        elif response.status >= 400:
                            error_text = await response.text()
                            raise APIError(f"API error {response.status}: {error_text}")
                        
                        response_data = await response.json()
                        content = response_data.get("content", [{}])[0].get("text", "")
                        
                        latency_ms = (time.time() - start_time) * 1000
                        
                        # Parse structured output if format specified
                        parsed_data = None
                        if request.response_format and content:
                            try:
                                # Try to parse JSON from content
                                json_str = self._extract_json(content)
                                if json_str:
                                    json_data = json.loads(json_str)
                                    parsed_data = request.response_format.model_validate(json_data)
                            except (json.JSONDecodeError, PydanticValidationError) as e:
                                raise ValidationError(f"Failed to parse LLM response: {e}")
                        
                        return LLMResponse(
                            content=content,
                            parsed_data=parsed_data,
                            latency_ms=latency_ms,
                            token_usage=response_data.get("usage", {})
                        )
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries:
                    raise APIError(f"API call failed after {self.max_retries} retries: {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
            except (RateLimitError, ValidationError):
                raise
            except Exception as e:
                if attempt == self.max_retries:
                    raise APIError(f"Unexpected error: {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def call_batch(self, requests: List[LLMRequest]) -> BatchResult:
        """Process multiple requests with concurrency control."""
        # Use semaphore to control concurrency and respect rate limits
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def bounded_call(req: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self.call_single(req)
        
        start_time = time.time()
        
        tasks = [bounded_call(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        failed_requests = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_requests.append((i, result))
            else:
                responses.append(result)
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return BatchResult(
            responses=responses,
            failed_requests=failed_requests,
            total_latency_ms=total_latency_ms,
            successful_count=len(responses),
            failed_count=len(failed_requests)
        )
    
    def _extract_json(self, content: str) -> Optional[str]:
        """Extract JSON from LLM response content."""
        # Try to find JSON in code blocks first
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find JSON object directly
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(0)
        
        return None


class LLMClient:
    """High-level client for LLM operations with built-in retry and error handling."""
    
    def __init__(
        self,
        provider: LLMProvider,
        default_retry_count: int = 3,
        default_retry_delay: float = 1.0,
        rate_limit_delay: float = 2.0
    ):
        self.provider = provider
        self.default_retry_count = default_retry_count
        self.default_retry_delay = default_retry_delay
        self.rate_limit_delay = rate_limit_delay
        
    async def call(
        self,
        prompt: str,
        response_format: Optional[Type[T]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retry_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[LLMResponse, T]:
        """Make a single LLM call with retry logic."""
        request = LLMRequest(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {}
        )
        
        retry_count = retry_count or self.default_retry_count
        
        for attempt in range(retry_count + 1):
            try:
                response = await self.provider.call_single(request)
                
                # Log the interaction
                logger.info(
                    "LLM call completed",
                    extra={
                        "prompt_length": len(prompt),
                        "response_length": len(response.content),
                        "latency_ms": response.latency_ms,
                        "attempt": attempt + 1,
                        "tokens": response.token_usage,
                        "has_structured_output": response.parsed_data is not None
                    }
                )
                
                # Return parsed data if available, otherwise full response
                return response.parsed_data if response.parsed_data else response
                
            except RateLimitError:
                logger.warning(f"Rate limit hit, waiting {self.rate_limit_delay}s before retry")
                await asyncio.sleep(self.rate_limit_delay)
                continue
            except ValidationError as e:
                if attempt == retry_count:
                    logger.error(f"Validation failed after {retry_count} retries: {e}")
                    raise
                logger.warning(f"Validation error on attempt {attempt + 1}, retrying: {e}")
                await asyncio.sleep(self.default_retry_delay * (attempt + 1))
            except Exception as e:
                if attempt == retry_count:
                    logger.error(f"LLM call failed after {retry_count} retries: {e}")
                    raise LLMError(f"Failed after {retry_count} retries: {e}")
                logger.warning(f"Error on attempt {attempt + 1}, retrying: {e}")
                await asyncio.sleep(self.default_retry_delay * (attempt + 1))
    
    async def call_batch(
        self,
        requests: List[tuple[str, Optional[Type[BaseModel]]]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """Make multiple LLM calls in batch."""
        llm_requests = [
            LLMRequest(
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=metadata or {}
            )
            for prompt, response_format in requests
        ]
        
        start_time = time.time()
        
        try:
            result = await self.provider.call_batch(llm_requests)
            
            logger.info(
                "Batch LLM call completed",
                extra={
                    "total_requests": len(requests),
                    "successful": result.successful_count,
                    "failed": result.failed_count,
                    "total_latency_ms": result.total_latency_ms,
                    "avg_latency_ms": result.total_latency_ms / len(requests) if requests else 0
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Batch LLM call failed: {e}")
            raise LLMError(f"Batch operation failed: {e}")


# Convenience function for creating LLM clients
def create_llm_client(
    provider_type: str = "mock",
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """Create an LLM client with the specified provider."""
    if provider_type == "mock":
        provider = MockLLMProvider(**kwargs)
    elif provider_type == "claude":
        if not api_key:
            raise ValueError("API key required for Claude provider")
        provider = ClaudeLLMProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return LLMClient(provider=provider)