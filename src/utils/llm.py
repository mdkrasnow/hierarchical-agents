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

try:
    import google.genai as genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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


class GeminiLLMProvider(LLMProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package not available. Install with: pip install google-genai")
        
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.client = genai.Client(api_key=api_key)
        
    async def call_single(self, request: LLMRequest) -> LLMResponse:
        """Make a single Gemini API call with retry logic."""
        start_time = time.time()
        
        # Prepare the generation config
        generation_config = {
            "temperature": request.temperature,
        }
        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        
        # If structured output is requested, add JSON schema instructions to prompt
        prompt = request.prompt
        if request.response_format:
            schema = request.response_format.model_json_schema()
            prompt += f"\n\nPlease respond with valid JSON that matches this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"
        
        for attempt in range(self.max_retries + 1):
            try:
                # Make the API call using the genai client
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(**generation_config)
                )
                
                # Extract content from response
                content = response.text if hasattr(response, 'text') else str(response)
                
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
                        raise ValidationError(f"Failed to parse Gemini response: {e}")
                
                # Extract token usage if available
                token_usage = {}
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    token_usage = {
                        "input_tokens": getattr(usage, 'prompt_token_count', 0),
                        "output_tokens": getattr(usage, 'candidates_token_count', 0),
                        "total_tokens": getattr(usage, 'total_token_count', 0)
                    }
                
                return LLMResponse(
                    content=content,
                    parsed_data=parsed_data,
                    latency_ms=latency_ms,
                    token_usage=token_usage
                )
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limiting
                if "rate limit" in error_str or "quota" in error_str:
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Gemini rate limit hit, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError("Gemini rate limit exceeded")
                
                # Check for API errors
                if "api" in error_str or "invalid" in error_str:
                    raise APIError(f"Gemini API error: {e}")
                
                # For other errors, retry with exponential backoff
                if attempt == self.max_retries:
                    raise APIError(f"Gemini API call failed after {self.max_retries} retries: {e}")
                
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def call_batch(self, requests: List[LLMRequest]) -> BatchResult:
        """Process multiple requests with concurrency control."""
        # Use semaphore to control concurrency and respect rate limits
        semaphore = asyncio.Semaphore(3)  # Lower concurrency for Gemini
        
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
        """Extract JSON from Gemini response content."""
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
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """Create an LLM client with the specified provider.
    
    If provider_type is None, will auto-select based on available environment variables:
    1. Gemini if GEMINI_API_KEY is set
    2. Claude if ANTHROPIC_API_KEY is set  
    3. OpenAI if OPENAI_API_KEY is set (when implemented)
    
    Requires a valid API key - no mock fallback.
    """
    import os
    
    # Auto-select provider based on environment if not specified
    if provider_type is None:
        if os.getenv("GEMINI_API_KEY"):
            provider_type = "gemini"
            api_key = api_key or os.getenv("GEMINI_API_KEY")
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider_type = "claude"
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):
            provider_type = "openai"
            api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(
                "No LLM API key found in environment. Please set one of:\n"
                "- GEMINI_API_KEY (recommended)\n"
                "- ANTHROPIC_API_KEY\n"
                "- OPENAI_API_KEY (when implemented)\n"
                "\nGet a Gemini API key at: https://aistudio.google.com/apikey"
            )
    
    # Use provided or environment API key if not specified
    if api_key is None:
        if provider_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif provider_type == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
    
    # Create the appropriate provider
    if provider_type == "claude":
        if not api_key:
            raise ValueError("API key required for Claude provider")
        provider = ClaudeLLMProvider(api_key=api_key, **kwargs)
    elif provider_type == "gemini":
        if not api_key:
            raise ValueError("API key required for Gemini provider")
        provider = GeminiLLMProvider(api_key=api_key, **kwargs)
    elif provider_type == "openai":
        # Note: OpenAI provider not yet implemented
        raise ValueError("OpenAI provider not yet implemented")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Supported: gemini, claude")
    
    return LLMClient(provider=provider)