"""Prompt template system with role/agent-type placeholders and template management."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Union

import yaml


class TemplateFormat(Enum):
    """Supported template formats."""
    STRING = "string"
    YAML = "yaml" 
    JSON = "json"


@dataclass
class PromptVariable:
    """Represents a variable in a prompt template."""
    name: str
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    variable_type: str = "str"
    examples: List[str] = field(default_factory=list)


@dataclass
class PromptTemplate:
    """Represents a prompt template with variables and metadata."""
    name: str
    template: str
    description: str = ""
    variables: List[PromptVariable] = field(default_factory=list)
    format: TemplateFormat = TemplateFormat.STRING
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        # Check for missing required variables
        variable_names = {var.name for var in self.variables}
        required_vars = {var.name for var in self.variables if var.required}
        provided_vars = set(kwargs.keys())
        
        missing_required = required_vars - provided_vars
        if missing_required:
            raise ValueError(f"Missing required variables: {missing_required}")
        
        # Add default values for missing optional variables
        render_vars = kwargs.copy()
        for var in self.variables:
            if var.name not in render_vars and var.default_value is not None:
                render_vars[var.name] = var.default_value
        
        # Validate that all provided variables are defined
        undefined_vars = provided_vars - variable_names
        if undefined_vars:
            # Warning but don't fail - allows for dynamic variables
            pass
        
        # Render using string.Template for safe substitution
        template = Template(self.template)
        try:
            return template.substitute(render_vars)
        except KeyError as e:
            raise ValueError(f"Template rendering failed: missing variable {e}")
    
    def get_variable_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all template variables."""
        return {
            var.name: {
                "description": var.description,
                "required": var.required,
                "type": var.variable_type,
                "default": var.default_value,
                "examples": var.examples
            }
            for var in self.variables
        }


class TemplateLoader(ABC):
    """Abstract base class for template loaders."""
    
    @abstractmethod
    async def load_template(self, template_name: str) -> PromptTemplate:
        """Load a template by name."""
        pass
    
    @abstractmethod
    async def list_templates(self) -> List[str]:
        """List all available template names."""
        pass


class FileTemplateLoader(TemplateLoader):
    """Load templates from filesystem."""
    
    def __init__(self, templates_dir: Union[str, Path]):
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_template(self, template_name: str) -> PromptTemplate:
        """Load a template from file."""
        # Try different file extensions
        for ext in [".yaml", ".yml", ".json", ".txt"]:
            template_path = self.templates_dir / f"{template_name}{ext}"
            if template_path.exists():
                return self._load_from_file(template_path)
        
        raise FileNotFoundError(f"Template '{template_name}' not found in {self.templates_dir}")
    
    async def list_templates(self) -> List[str]:
        """List all available templates."""
        templates = []
        for ext in [".yaml", ".yml", ".json", ".txt"]:
            for template_path in self.templates_dir.glob(f"*{ext}"):
                template_name = template_path.stem
                if template_name not in templates:
                    templates.append(template_name)
        return sorted(templates)
    
    def _load_from_file(self, template_path: Path) -> PromptTemplate:
        """Load template from a specific file."""
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if template_path.suffix in [".yaml", ".yml"]:
            return self._load_yaml_template(template_path.stem, content)
        elif template_path.suffix == ".json":
            return self._load_json_template(template_path.stem, content)
        else:
            return self._load_string_template(template_path.stem, content)
    
    def _load_yaml_template(self, name: str, content: str) -> PromptTemplate:
        """Load template from YAML format."""
        data = yaml.safe_load(content)
        
        variables = []
        if "variables" in data:
            for var_data in data["variables"]:
                variables.append(PromptVariable(**var_data))
        
        return PromptTemplate(
            name=name,
            template=data["template"],
            description=data.get("description", ""),
            variables=variables,
            format=TemplateFormat.YAML,
            tags=data.get("tags", []),
            version=data.get("version", "1.0")
        )
    
    def _load_json_template(self, name: str, content: str) -> PromptTemplate:
        """Load template from JSON format."""
        data = json.loads(content)
        
        variables = []
        if "variables" in data:
            for var_data in data["variables"]:
                variables.append(PromptVariable(**var_data))
        
        return PromptTemplate(
            name=name,
            template=data["template"],
            description=data.get("description", ""),
            variables=variables,
            format=TemplateFormat.JSON,
            tags=data.get("tags", []),
            version=data.get("version", "1.0")
        )
    
    def _load_string_template(self, name: str, content: str) -> PromptTemplate:
        """Load template from plain string format."""
        return PromptTemplate(
            name=name,
            template=content,
            description=f"Plain text template: {name}",
            format=TemplateFormat.STRING
        )


class InMemoryTemplateLoader(TemplateLoader):
    """In-memory template loader for testing and dynamic templates."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
    
    def add_template(self, template: PromptTemplate):
        """Add a template to the in-memory store."""
        self.templates[template.name] = template
    
    async def load_template(self, template_name: str) -> PromptTemplate:
        """Load a template from memory."""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    async def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())


class TemplateManager:
    """High-level template management with caching and multiple loaders."""
    
    def __init__(self, default_loader: Optional[TemplateLoader] = None):
        self.loaders: Dict[str, TemplateLoader] = {}
        self.template_cache: Dict[str, PromptTemplate] = {}
        self.default_loader = default_loader or InMemoryTemplateLoader()
        
        # Built-in templates for common agent patterns
        self._init_builtin_templates()
    
    def add_loader(self, name: str, loader: TemplateLoader):
        """Add a template loader."""
        self.loaders[name] = loader
    
    async def get_template(self, template_name: str, loader_name: Optional[str] = None) -> PromptTemplate:
        """Get a template by name, with caching."""
        cache_key = f"{loader_name or 'default'}:{template_name}"
        
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
        loader = self.loaders.get(loader_name) if loader_name else self.default_loader
        if not loader:
            raise ValueError(f"Loader '{loader_name}' not found")
        
        template = await loader.load_template(template_name)
        self.template_cache[cache_key] = template
        return template
    
    async def render_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        loader_name: Optional[str] = None
    ) -> str:
        """Render a template with variables."""
        template = await self.get_template(template_name, loader_name)
        return template.render(**variables)
    
    async def list_all_templates(self) -> Dict[str, List[str]]:
        """List templates from all loaders."""
        all_templates = {}
        
        # Default loader
        all_templates["default"] = await self.default_loader.list_templates()
        
        # Named loaders
        for name, loader in self.loaders.items():
            all_templates[name] = await loader.list_templates()
        
        return all_templates
    
    def clear_cache(self):
        """Clear the template cache."""
        self.template_cache.clear()
    
    def _init_builtin_templates(self):
        """Initialize built-in templates for common patterns."""
        if isinstance(self.default_loader, InMemoryTemplateLoader):
            # Base agent system prompt
            system_prompt = PromptTemplate(
                name="system_prompt",
                template="""You are $agent_type, a specialized AI agent in a hierarchical agent system.

Your role: $role_description

Key responsibilities:
$responsibilities

Context: $context

Please follow these guidelines:
- Be precise and thorough in your analysis
- Provide structured output when requested  
- Acknowledge limitations and uncertainties
- Collaborate effectively with other agents

$additional_instructions""",
                description="Standard system prompt template for agents",
                variables=[
                    PromptVariable("agent_type", "Type of agent (e.g., 'a code reviewer', 'an implementation specialist')", True),
                    PromptVariable("role_description", "Detailed description of the agent's role", True),
                    PromptVariable("responsibilities", "List of key responsibilities", True),
                    PromptVariable("context", "Current context or task information", True),
                    PromptVariable("additional_instructions", "Any additional specific instructions", False, "")
                ],
                tags=["system", "agent", "base"]
            )
            
            # Task delegation template
            task_delegation = PromptTemplate(
                name="task_delegation",
                template="""# Task Delegation

**Task ID**: $task_id
**Assigned to**: $agent_type
**Priority**: $priority

## Task Description
$task_description

## Context
$context

## Requirements
$requirements

## Expected Output
$expected_output

## Deadline
$deadline

## Additional Notes
$additional_notes""",
                description="Template for delegating tasks to sub-agents",
                variables=[
                    PromptVariable("task_id", "Unique identifier for the task", True),
                    PromptVariable("agent_type", "Type of agent to delegate to", True),
                    PromptVariable("priority", "Task priority level", True),
                    PromptVariable("task_description", "Detailed task description", True),
                    PromptVariable("context", "Relevant context for the task", True),
                    PromptVariable("requirements", "Specific requirements and constraints", True),
                    PromptVariable("expected_output", "Description of expected output format", True),
                    PromptVariable("deadline", "Task deadline", False, "No specific deadline"),
                    PromptVariable("additional_notes", "Any additional notes or instructions", False, "")
                ],
                tags=["delegation", "task", "coordination"]
            )
            
            # Result synthesis template
            result_synthesis = PromptTemplate(
                name="result_synthesis",
                template="""# Result Synthesis

Synthesize the following results from multiple agents:

## Agent Results
$agent_results

## Synthesis Instructions
$synthesis_instructions

## Output Requirements
- Identify consensus areas
- Highlight disagreements or conflicts
- Provide unified recommendations
- Note any gaps or missing information

$output_format

## Quality Criteria
$quality_criteria""",
                description="Template for synthesizing results from multiple agents",
                variables=[
                    PromptVariable("agent_results", "Results from different agents", True),
                    PromptVariable("synthesis_instructions", "Specific instructions for synthesis", True),
                    PromptVariable("output_format", "Required output format specification", False, "Provide output in clear, structured format."),
                    PromptVariable("quality_criteria", "Criteria for evaluating synthesis quality", False, "Ensure accuracy, completeness, and clarity.")
                ],
                tags=["synthesis", "coordination", "results"]
            )
            
            # Add built-in templates to default loader
            self.default_loader.add_template(system_prompt)
            self.default_loader.add_template(task_delegation)
            self.default_loader.add_template(result_synthesis)


# Global template manager instance
_global_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Get the global template manager instance."""
    global _global_template_manager
    if _global_template_manager is None:
        _global_template_manager = TemplateManager()
    return _global_template_manager


def set_template_manager(manager: TemplateManager):
    """Set the global template manager instance."""
    global _global_template_manager
    _global_template_manager = manager


async def render_prompt(template_name: str, **kwargs) -> str:
    """Convenience function to render a prompt template."""
    manager = get_template_manager()
    return await manager.render_template(template_name, kwargs)