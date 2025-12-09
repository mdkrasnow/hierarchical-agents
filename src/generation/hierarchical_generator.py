"""
Answer generation using hierarchical agents.

Provides answer generation by leveraging the existing hierarchical agent
system for specialized question types that benefit from multi-agent coordination.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

from agents.base import BaseAgent, AgentConfig
from agents.evaluation import DanielsonEvaluationAgent
from agents.danielson import DanielsonSpecificAgent
from utils.llm import LLMClient

from .answer_generator import (
    AnswerGenerator,
    GenerationConfig,
    GenerationResult,
    GenerationStrategy
)


class HierarchicalGenerator(AnswerGenerator):
    """
    Answer generator using hierarchical agents for complex, specialized questions.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[GenerationConfig] = None,
        available_agents: Optional[Dict[str, BaseAgent]] = None
    ):
        """
        Initialize hierarchical generator.
        
        Args:
            llm_client: LLM client for agent operations
            config: Generation configuration
            available_agents: Pre-configured agents to use
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.available_agents = available_agents or {}
        
        # Initialize default agents if LLM client provided
        if self.llm_client:
            self._initialize_default_agents()
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this generation strategy."""
        return "hierarchical_agent"
    
    @property
    def supported_question_types(self) -> List[str]:
        """Return list of question types this generator supports well."""
        return [
            "evaluation-analysis",
            "danielson-evaluation",
            "educational-assessment", 
            "teacher-performance",
            "data-analysis",
            "complex-reasoning",
            "multi-criteria-analysis",
            "professional-development"
        ]
    
    def _initialize_default_agents(self):
        """Initialize default agents with the LLM client."""
        if "evaluation_agent" not in self.available_agents:
            agent_config = AgentConfig(llm_temperature=self.config.temperature)
            self.available_agents["evaluation_agent"] = DanielsonEvaluationAgent(
                llm_client=self.llm_client,
                config=agent_config
            )
        
        if "danielson_agent" not in self.available_agents:
            agent_config = AgentConfig(llm_temperature=self.config.temperature)
            self.available_agents["danielson_agent"] = DanielsonSpecificAgent(
                llm_client=self.llm_client,
                config=agent_config
            )
    
    async def generate_answer(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        config_override: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate an answer using hierarchical agents."""
        start_time = time.time()
        
        # Merge configuration
        effective_config = self._merge_configs(config_override)
        context = context or {}
        
        # Create initial result
        result = self._create_base_result(start_time=start_time)
        result.add_reasoning_step("Starting hierarchical agent-based generation")
        
        try:
            # Analyze question to determine best agent strategy
            agent_strategy = await self._analyze_question_for_agents(
                question, context, effective_config, result
            )
            
            if not agent_strategy:
                result.set_error("No suitable agent strategy found for this question")
                return result
            
            # Generate answer using selected strategy
            if agent_strategy["type"] == "single_agent":
                answer = await self._generate_with_single_agent(
                    question, context, effective_config, result, agent_strategy
                )
            elif agent_strategy["type"] == "multi_agent":
                answer = await self._generate_with_multi_agent(
                    question, context, effective_config, result, agent_strategy
                )
            else:
                result.set_error(f"Unknown agent strategy type: {agent_strategy['type']}")
                return result
            
            # Validate answer if enabled
            if effective_config.enable_validation:
                validation = await self.validate_answer(question, answer, context)
                result.confidence_score = validation.get("quality_score", 0.5)
                result.estimated_quality = self._estimate_quality_tier(validation.get("quality_score", 0.5))
                
                if not validation["is_valid"]:
                    result.add_warning("Generated answer failed validation")
                    for issue in validation.get("issues", []):
                        result.add_warning(f"Validation issue: {issue}")
            
            # Finalize result
            result.answer = answer
            result.generation_time_ms = (time.time() - start_time) * 1000
            result.add_reasoning_step(f"Hierarchical generation completed with {len(answer.split())} words")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hierarchical generation failed: {e}")
            result.set_error(str(e))
            result.generation_time_ms = (time.time() - start_time) * 1000
            return result
    
    async def _analyze_question_for_agents(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult
    ) -> Optional[Dict[str, Any]]:
        """Analyze question to determine the best agent strategy."""
        result.add_reasoning_step("Analyzing question to select appropriate agents")
        
        question_lower = question.lower()
        context_str = str(context).lower() if context else ""
        
        # Danielson evaluation questions
        danielson_keywords = [
            "danielson", "evaluation", "teacher", "domain", "rubric",
            "professional practice", "classroom management", "instruction"
        ]
        if any(keyword in question_lower for keyword in danielson_keywords):
            result.add_reasoning_step("Question identified as Danielson evaluation type")
            return {
                "type": "single_agent",
                "primary_agent": "danielson_agent",
                "reasoning": "Question contains Danielson evaluation keywords"
            }
        
        # General evaluation questions
        evaluation_keywords = [
            "evaluate", "assess", "analyze", "summarize", "performance",
            "strengths", "concerns", "evidence", "observation"
        ]
        if any(keyword in question_lower for keyword in evaluation_keywords):
            result.add_reasoning_step("Question identified as general evaluation type")
            return {
                "type": "single_agent", 
                "primary_agent": "evaluation_agent",
                "reasoning": "Question contains evaluation keywords"
            }
        
        # Complex multi-faceted questions
        complex_keywords = [
            "compare", "synthesize", "integrate", "multiple", "comprehensive",
            "holistic", "overall", "district", "school-wide"
        ]
        if any(keyword in question_lower for keyword in complex_keywords):
            result.add_reasoning_step("Question identified as complex multi-agent type")
            return {
                "type": "multi_agent",
                "primary_agent": "evaluation_agent",
                "supporting_agents": ["danielson_agent"],
                "coordination": "synthesis",
                "reasoning": "Question requires multiple perspectives"
            }
        
        # Check if specific agent types are requested in config
        if config.agent_types:
            requested_agent = config.agent_types[0]  # Use first requested agent
            if requested_agent in self.available_agents:
                result.add_reasoning_step(f"Using requested agent: {requested_agent}")
                return {
                    "type": "single_agent",
                    "primary_agent": requested_agent,
                    "reasoning": "Agent explicitly requested in config"
                }
        
        # Default to evaluation agent for unknown questions
        if "evaluation_agent" in self.available_agents:
            result.add_reasoning_step("Defaulting to evaluation agent for unknown question type")
            return {
                "type": "single_agent",
                "primary_agent": "evaluation_agent", 
                "reasoning": "Default agent for general questions"
            }
        
        result.add_warning("No suitable agent found for question")
        return None
    
    async def _generate_with_single_agent(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult,
        strategy: Dict[str, Any]
    ) -> str:
        """Generate answer using a single agent."""
        agent_name = strategy["primary_agent"]
        result.add_reasoning_step(f"Using single agent: {agent_name}")
        
        if agent_name not in self.available_agents:
            raise ValueError(f"Agent '{agent_name}' not available")
        
        agent = self.available_agents[agent_name]
        
        # Prepare agent input based on agent type
        if isinstance(agent, (DanielsonEvaluationAgent, DanielsonSpecificAgent)):
            # These agents expect structured input - adapt the question
            agent_input = self._adapt_question_for_danielson_agent(question, context)
        else:
            # Generic agent input
            agent_input = {
                "question": question,
                "context": context,
                "config": config.generation_context
            }
        
        # Execute agent
        agent_result = await agent.execute_with_tracking(**agent_input)
        
        # Track agent execution
        result.llm_calls_made += getattr(agent_result, 'llm_calls', 0)
        result.agent_outputs[agent_name] = agent_result.data if agent_result.success else agent_result.error
        
        if not agent_result.success:
            raise RuntimeError(f"Agent {agent_name} failed: {agent_result.error}")
        
        # Extract answer from agent result
        answer = self._extract_answer_from_agent_result(agent_result, agent_name)
        result.add_reasoning_step(f"Agent {agent_name} generated {len(answer.split())} word answer")
        
        return answer
    
    async def _generate_with_multi_agent(
        self,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult,
        strategy: Dict[str, Any]
    ) -> str:
        """Generate answer using multiple coordinated agents."""
        primary_agent_name = strategy["primary_agent"]
        supporting_agent_names = strategy.get("supporting_agents", [])
        coordination_method = strategy.get("coordination", "synthesis")
        
        result.add_reasoning_step(f"Using multi-agent with primary: {primary_agent_name}, supporting: {supporting_agent_names}")
        
        # Execute all agents concurrently
        agent_tasks = {}
        
        # Primary agent
        if primary_agent_name in self.available_agents:
            agent_tasks["primary"] = self._execute_agent(
                self.available_agents[primary_agent_name], 
                question, context, config, "primary"
            )
        
        # Supporting agents
        for i, agent_name in enumerate(supporting_agent_names):
            if agent_name in self.available_agents:
                agent_tasks[f"supporting_{i}"] = self._execute_agent(
                    self.available_agents[agent_name],
                    question, context, config, f"supporting_{i}"
                )
        
        # Wait for all agents to complete
        agent_results = await asyncio.gather(
            *agent_tasks.values(),
            return_exceptions=True
        )
        
        # Process agent results
        successful_results = {}
        for i, (task_name, agent_result) in enumerate(zip(agent_tasks.keys(), agent_results)):
            if isinstance(agent_result, Exception):
                result.add_warning(f"Agent {task_name} failed: {agent_result}")
                continue
            
            successful_results[task_name] = agent_result
            result.agent_outputs[task_name] = agent_result.get("data", {})
        
        if not successful_results:
            raise RuntimeError("All agents failed to execute")
        
        # Coordinate results based on strategy
        if coordination_method == "synthesis":
            answer = await self._synthesize_agent_answers(
                successful_results, question, context, config, result
            )
        else:
            # Default: use primary agent answer or first successful
            primary_result = successful_results.get("primary")
            if primary_result:
                answer = self._extract_answer_from_result_dict(primary_result)
            else:
                first_result = next(iter(successful_results.values()))
                answer = self._extract_answer_from_result_dict(first_result)
        
        result.coordination_details = {
            "method": coordination_method,
            "agents_used": list(successful_results.keys()),
            "coordination_strategy": strategy
        }
        
        return answer
    
    async def _execute_agent(
        self,
        agent: BaseAgent,
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        agent_role: str
    ) -> Dict[str, Any]:
        """Execute a single agent and return structured result."""
        # Prepare agent input
        if isinstance(agent, (DanielsonEvaluationAgent, DanielsonSpecificAgent)):
            agent_input = self._adapt_question_for_danielson_agent(question, context)
        else:
            agent_input = {
                "question": question,
                "context": context,
                "config": config.generation_context
            }
        
        # Execute agent
        agent_result = await agent.execute_with_tracking(**agent_input)
        
        return {
            "success": agent_result.success,
            "data": agent_result.data,
            "error": agent_result.error,
            "metadata": getattr(agent_result, 'metadata', {}),
            "agent_type": agent.agent_type,
            "role": agent_role
        }
    
    async def _synthesize_agent_answers(
        self,
        agent_results: Dict[str, Dict[str, Any]],
        question: str,
        context: Dict[str, Any],
        config: GenerationConfig,
        result: GenerationResult
    ) -> str:
        """Synthesize answers from multiple agents."""
        result.add_reasoning_step("Synthesizing answers from multiple agents")
        
        # Extract individual answers
        individual_answers = []
        for agent_name, agent_result in agent_results.items():
            answer = self._extract_answer_from_result_dict(agent_result)
            if answer:
                individual_answers.append(f"[{agent_name}] {answer}")
        
        if not individual_answers:
            raise RuntimeError("No valid answers from any agent")
        
        # Simple synthesis: combine and summarize
        synthesis_prompt = f"""
Question: {question}

Individual agent responses:
{chr(10).join(individual_answers)}

Please synthesize these responses into a single, coherent answer that integrates the key insights from each perspective. The final answer should be comprehensive yet concise.

Synthesized Answer:"""
        
        # Use LLM client for synthesis if available
        if self.llm_client:
            response = await self.llm_client.call(
                prompt=synthesis_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            if hasattr(response, 'content'):
                synthesized = response.content.strip()
            else:
                synthesized = str(response).strip()
            
            result.llm_calls_made += 1
            result.add_reasoning_step("Synthesized multiple agent perspectives using LLM")
        else:
            # Fallback: simple concatenation
            synthesized = "\n\nBased on multiple analytical perspectives:\n\n" + "\n\n".join([
                answer.split("] ", 1)[1] if "] " in answer else answer 
                for answer in individual_answers
            ])
            result.add_reasoning_step("Synthesized using simple concatenation")
        
        return synthesized
    
    def _adapt_question_for_danielson_agent(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt a general question for Danielson-specific agents."""
        # Import here to avoid circular imports
        try:
            from agents.evaluation import EvaluationInput
            from models.test_models import DanielsonEvaluation
        except ImportError:
            # Fallback if models not available
            return {"question": question, "context": context}
        
        from datetime import datetime
        import uuid
        
        # Create a mock evaluation structure if none provided
        mock_evaluation_data = {
            "id": str(uuid.uuid4()),
            "teacher_name": context.get("teacher_name", "Sample Teacher"),
            "school_name": context.get("school_name", "Sample School"),
            "evaluator": "System Generated",
            "framework_id": "danielson_2023",
            "is_informal": True,
            "created_at": datetime.now(),
            "low_inference_notes": question,  # Use question as notes
            "evaluation": context.get("evaluation_data", {
                "domains": {
                    "I-A": {"score": 3, "notes": "Based on question context"},
                    "II-A": {"score": 3, "notes": "Generated for analysis"},
                    "III-A": {"score": 3, "notes": "Generated for analysis"}
                }
            })
        }
        
        org_config = context.get("organization_config", {
            "version": "1.0",
            "framework": "danielson_2023",
            "domains": {
                "I-A": {"name": "Knowledge of Content and Pedagogy", "green": 3, "yellow": 2, "red": 1},
                "II-A": {"name": "Environment of Respect and Rapport", "green": 3, "yellow": 2, "red": 1},
                "III-A": {"name": "Communicating with Students", "green": 3, "yellow": 2, "red": 1}
            },
            "global_thresholds": {
                "exemplar_teacher": 3.5,
                "proficient_teacher": 2.8
            }
        })
        
        try:
            # Create proper evaluation input object
            mock_evaluation = DanielsonEvaluation(**mock_evaluation_data)
            evaluation_input = EvaluationInput(
                evaluation_data=mock_evaluation,
                organization_config=org_config,
                analysis_focus="comprehensive"
            )
            
            return {"evaluation_input": evaluation_input}
            
        except Exception as e:
            # Fallback if object creation fails
            self.logger.warning(f"Failed to create evaluation input: {e}")
            return {"question": question, "context": context}
    
    def _extract_answer_from_agent_result(
        self,
        agent_result: Any,
        agent_name: str
    ) -> str:
        """Extract answer text from agent result."""
        if not hasattr(agent_result, 'data') or not agent_result.data:
            return f"Agent {agent_name} completed but returned no data."
        
        data = agent_result.data
        
        # Try different ways to extract meaningful answer text
        if isinstance(data, dict):
            # Look for common answer fields
            for key in ['answer', 'result', 'summary', 'analysis', 'evaluation_summary']:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        # Try to create readable summary from dict
                        return self._dict_to_readable_text(value)
            
            # Fallback: convert entire dict to readable format
            return self._dict_to_readable_text(data)
        
        elif isinstance(data, str):
            return data
        
        else:
            return str(data)
    
    def _extract_answer_from_result_dict(self, result_dict: Dict[str, Any]) -> str:
        """Extract answer from result dictionary."""
        if not result_dict.get("success", False):
            return f"Agent failed: {result_dict.get('error', 'Unknown error')}"
        
        return self._extract_answer_from_agent_result(
            type('MockResult', (), result_dict)(),
            result_dict.get("role", "agent")
        )
    
    def _dict_to_readable_text(self, data: Dict[str, Any]) -> str:
        """Convert dictionary data to readable text."""
        if not data:
            return "No data available."
        
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"{key}: {', '.join(str(item) for item in value)}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
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