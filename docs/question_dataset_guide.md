# Question Dataset Guide

This guide explains how to use the evaluation question dataset system for the hierarchical multi-agent scoring framework.

## Overview

The question dataset system provides a structured way to manage evaluation questions used to assess the quality of AI-generated responses. It supports:

- **Structured question format** with metadata for comprehensive evaluation
- **Flexible filtering** by category, difficulty, tags, and rubric focus
- **Balanced sampling** for creating test sets
- **Statistical analysis** of dataset composition
- **Search functionality** for finding relevant questions

## Question Structure

Each question in the dataset contains the following fields:

### Required Fields

- **`id`**: Unique identifier (e.g., "hma-001", "sql-003")
- **`prompt`**: The actual question or task description
- **`category`**: Main topic area (hierarchical-agents, supabase-sql, general-writing)
- **`tags`**: List of descriptive tags for filtering and search
- **`expected_key_points`**: List of points that should be covered in a good answer
- **`style_requirements`**: Dictionary specifying format, tone, length, and other style expectations
- **`difficulty`**: Complexity level (beginner, intermediate, advanced, expert)
- **`rubric_focus`**: List of rubric dimensions this question is designed to test
- **`created_date`**: ISO format date when question was added

### Optional Fields

- **`description`**: Additional context or notes about the question
- **`notes`**: Internal notes for question maintainers
- **`last_modified`**: Date of last modification

### Example Question

```json
{
  "id": "hma-001",
  "prompt": "Explain how to design a hierarchical multi-agent system for complex problem-solving. Include the key architectural components, communication patterns, and coordination mechanisms.",
  "category": "hierarchical-agents",
  "tags": ["architecture", "design-patterns", "coordination", "multi-agent-systems"],
  "expected_key_points": [
    "hierarchical structure definition",
    "agent roles and responsibilities", 
    "communication protocols",
    "task decomposition strategies",
    "coordination mechanisms",
    "failure handling",
    "performance considerations"
  ],
  "style_requirements": {
    "format": "structured_guide",
    "tone": "technical_professional", 
    "length": "detailed",
    "include_examples": true
  },
  "difficulty": "advanced",
  "rubric_focus": ["coverage", "detail_specificity"],
  "created_date": "2024-12-08"
}
```

## Categories

The dataset includes three main categories:

### Hierarchical-Agents
**Focus**: Multi-agent system design, coordination, and implementation
- **Primary Rubric Focus**: Coverage, Detail & Specificity
- **Common Tags**: architecture, coordination, communication, fault-tolerance
- **Difficulty Range**: Intermediate to Expert

### Supabase-SQL  
**Focus**: Database design, SQL optimization, and Supabase features
- **Primary Rubric Focus**: Detail & Specificity, Structure & Coherence  
- **Common Tags**: schema-design, performance, RLS, real-time, analytics
- **Difficulty Range**: Beginner to Advanced

### General-Writing
**Focus**: Technical writing, documentation, and communication
- **Primary Rubric Focus**: Style & Tone, Structure & Coherence
- **Common Tags**: documentation, communication, best-practices, clarity
- **Difficulty Range**: Beginner to Advanced

## Using the Dataset

### Basic Loading

```python
from scoring import QuestionLoader, load_all_questions

# Load all questions
loader = QuestionLoader()
all_questions = loader.get_all_questions()

# Or use the convenience function
all_questions = load_all_questions()
```

### Filtering Questions

```python
# Filter by category
hma_questions = loader.get_hierarchical_agents_questions()
sql_questions = loader.get_supabase_sql_questions() 
writing_questions = loader.get_writing_questions()

# Filter by difficulty
beginner_questions = loader.get_beginner_questions()
expert_questions = loader.get_expert_questions()

# Filter by tags
architecture_questions = loader.get_questions_by_tag("architecture")
performance_questions = loader.get_questions_by_tag("performance")

# Filter by rubric focus
coverage_questions = loader.get_coverage_focused_questions()
detail_questions = loader.get_detail_focused_questions()

# Advanced filtering
filtered = loader.filter_questions(
    category="hierarchical-agents",
    difficulty="advanced", 
    tags=["coordination", "fault-tolerance"]
)
```

### Sampling Questions

```python
# Random sample
sample = loader.dataset.sample_questions(5)

# Balanced sample across categories
balanced = loader.sample_balanced_questions(n_per_category=3)

# Sample with difficulty distribution
distributed = loader.sample_by_difficulty_distribution(
    total_questions=10,
    beginner_pct=0.2,
    intermediate_pct=0.4, 
    advanced_pct=0.3,
    expert_pct=0.1
)

# Create test set
test_set = loader.create_test_set(n_questions=8)
```

### Searching Questions

```python
from scoring.loaders import search_questions

# Search for questions containing specific terms
coordination_questions = search_questions("coordination")
database_questions = search_questions("database")
```

## Command-Line Interface

Use the `list_questions.py` script to explore the dataset:

```bash
# Show all questions (brief format)
python scripts/list_questions.py --all

# Show dataset statistics  
python scripts/list_questions.py --stats

# Filter by category
python scripts/list_questions.py --category hierarchical-agents

# Filter by difficulty and show detailed format
python scripts/list_questions.py --difficulty advanced --format detailed

# Sample questions
python scripts/list_questions.py --sample 5

# Search for specific topics
python scripts/list_questions.py --search "coordination" --format json

# Show specific question
python scripts/list_questions.py --id hma-001 --format detailed

# List available options
python scripts/list_questions.py --list-categories
python scripts/list_questions.py --list-tags
python scripts/list_questions.py --list-difficulties
```

## Dataset Statistics

Get comprehensive statistics about the dataset:

```python
stats = loader.get_dataset_statistics()
print(f"Total questions: {stats['total_questions']}")
print(f"Categories: {stats['categories']}")
print(f"Tag distribution: {stats['tags']}")

# Or print a formatted summary
loader.print_dataset_summary()
```

## Rubric Alignment

The questions are designed to stress different dimensions of the evaluation rubric:

### Coverage Questions (30% weight)
- Focus on comprehensive topic coverage
- Test breadth of knowledge
- Emphasize completeness of responses
- **Examples**: System design, comprehensive guides

### Detail & Specificity Questions (25% weight)  
- Require concrete examples and precise information
- Test depth of technical knowledge
- Emphasize actionable content
- **Examples**: Implementation details, optimization techniques

### Structure & Coherence Questions (20% weight)
- Emphasize logical organization
- Test clarity of presentation  
- Focus on information flow
- **Examples**: Tutorials, step-by-step guides

### Style & Tone Questions (15% weight)
- Test writing quality and professionalism
- Emphasize appropriate communication
- Focus on readability and engagement
- **Examples**: Documentation, communication strategies  

### Instruction Following Questions (10% weight)
- Test adherence to specific requirements
- Include format constraints
- Emphasize following guidelines precisely
- **Examples**: Specific output formats, word limits

## Best Practices

### Question Selection for Evaluation

1. **Balanced Representation**: Include questions from all categories
2. **Difficulty Distribution**: Mix difficulty levels appropriately for your evaluation goals
3. **Rubric Coverage**: Ensure questions test all rubric dimensions
4. **Relevance**: Select questions relevant to your specific use case

### Creating New Questions

When adding new questions to the dataset:

1. **Follow the Schema**: Include all required fields
2. **Be Specific**: Make expected key points concrete and measurable
3. **Set Clear Style Requirements**: Specify format, tone, and length expectations
4. **Tag Consistently**: Use existing tags where possible for better filtering
5. **Test the Question**: Ensure it can effectively discriminate between response quality levels

### Maintaining Quality

1. **Regular Review**: Periodically review questions for relevance and clarity
2. **Update Tags**: Keep tag taxonomy consistent and meaningful  
3. **Monitor Usage**: Track which questions are most/least effective
4. **Gather Feedback**: Collect input from evaluators about question quality

## Dataset File Format

Questions are stored in JSONL (JSON Lines) format:

- **Location**: `data/questions.jsonl`
- **Format**: One JSON object per line
- **Encoding**: UTF-8
- **Schema**: Each line follows the Question schema described above

### Categories Configuration

Additional metadata is stored in:

- **Location**: `configs/question_categories.yaml`
- **Contents**: Category definitions, tag taxonomy, style templates
- **Purpose**: Provides structure and guidelines for dataset organization

## Integration with Scoring System

The question dataset integrates with the broader scoring system:

1. **Rubric Alignment**: Questions are tagged with their primary rubric focus
2. **Scoring Context**: Expected key points guide coverage evaluation
3. **Style Requirements**: Used for instruction following assessment
4. **Filtering Support**: Enables targeted evaluation of specific capabilities

## Extending the Dataset

To add new questions:

1. **Add to JSONL file**: Append new questions to `data/questions.jsonl`
2. **Update categories**: If needed, update `configs/question_categories.yaml`
3. **Validate**: Use the loader to ensure proper formatting
4. **Test**: Verify filtering and search functionality works

```python
# Example: Adding a new question programmatically
from scoring import Question, QuestionDataset

new_question = Question(
    id="new-001",
    prompt="Your question here...",
    category="appropriate-category", 
    tags=["relevant", "tags"],
    expected_key_points=["point1", "point2"],
    style_requirements={"format": "guide", "tone": "technical"},
    difficulty="intermediate",
    rubric_focus=["coverage"],
    created_date="2024-12-08"
)

# Load existing dataset and add question
loader = QuestionLoader()
dataset = loader.dataset
dataset.add_question(new_question)

# Save updated dataset
dataset.save_to_file("data/questions.jsonl")
```

## Troubleshooting

### Common Issues

**Loading Errors**
- Check file paths are correct
- Ensure JSON syntax is valid in JSONL file
- Verify all required fields are present

**Filtering Returns Empty Results**  
- Check filter values match exactly (case-sensitive for some fields)
- Use `--list-categories`, `--list-tags` to see available values
- Try broader filters first, then narrow down

**Performance Issues**
- For large datasets, consider implementing caching
- Use specific filters rather than loading all questions first
- Consider database storage for very large question sets

### Getting Help

1. **Check Examples**: Review the example usage in this guide
2. **Use CLI Help**: Run `python scripts/list_questions.py --help`
3. **Inspect Code**: The modules are well-documented with docstrings
4. **Test with Samples**: Start with `--sample 5` to test functionality

---

This question dataset system provides a robust foundation for evaluating AI-generated responses across multiple dimensions and use cases. The flexible filtering and sampling capabilities enable targeted evaluation while the structured format ensures consistency and reproducibility.