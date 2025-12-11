# Multi-Agent Prompt Optimization System - Implementation Plan

**Generated:** 2025-12-10  
**Purpose:** Implement an automated prompt evaluation and optimization system using multi-agent review and iterative refinement  
**Status:** Planning Phase

---

## Executive Summary

This document outlines the implementation plan for building a **multi-agent prompt optimization system** that automatically evaluates agent performance on test datasets, identifies prompt weaknesses through multi-agent debate, and iteratively refines prompts to improve system performance.

The system will function as an **automated prompt backpropagation mechanism**, analogous to gradient descent in neural networks, but operating on natural language prompts through multi-agent reasoning and synthesis.

---

## 1. System Overview

### 1.1 Core Concept

Create a feedback loop system that:
1. **Forward Pass**: Evaluates current prompts on test dataset → generates answers → scores performance
2. **Multi-Agent Review**: Analyzes failures and successes → debates improvements → generates prompt modification suggestions
3. **Synthesis**: Aggregates all modification suggestions → creates coherent prompt updates
4. **Backward Pass**: Applies prompt modifications → re-evaluates performance → measures improvement
5. **Iteration**: Repeats cycle until performance threshold met or max iterations reached

### 1.2 Relationship to Existing Systems

**Builds upon:**
- Existing multi-critic evaluation system (`src/critics/`)
- Multi-agent debate framework (`.claude/commands/debate-solve.md`)
- Template/prompt management system (`src/agents/templates.py`)
- Calibration infrastructure (`src/eval/calibration.py`)
- Test dataset management (`src/scoring/dataset.py`)

**New components:**
- Prompt evaluation orchestrator
- Performance-to-prompt-modification mapping agents
- Prompt modification synthesis system
- Iterative optimization controller
- Improvement tracking and metrics

---

## 2. Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Prompt Optimization Controller                  │
│  (Manages optimization loop, tracks iterations, metrics)    │
└────────────┬───────────────────────────────────┬────────────┘
             │                                   │
      ┌──────▼──────┐                    ┌──────▼──────────┐
      │  Forward     │                    │  Backward       │
      │  Pass        │                    │  Pass           │
      │  Evaluator   │                    │  Modifier       │
      └──────┬───────┘                    └─────▲───────────┘
             │                                   │
    ┌────────▼────────────┐              ┌──────┴──────────┐
    │ Test Dataset        │              │ Prompt          │
    │ Execution           │              │ Synthesis       │
    │ - Run questions     │              │ Agent           │
    │ - Generate answers  │              │                 │
    │ - Score results     │              └─────▲───────────┘
    └────────┬────────────┘                    │
             │                          ┌──────┴──────────┐
    ┌────────▼────────────┐            │ Multi-Agent      │
    │ Performance         │            │ Debate System    │
    │ Analysis            │───────────▶│ (3 rounds)       │
    │ - Per-question      │            │ - Performance    │
    │ - Per-prompt        │            │   Reviewer       │
    │ - Per-dimension     │            │ - Modification   │
    └─────────────────────┘            │   Proposer       │
                                       │ - Synthesis      │
                                       └──────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 **Prompt Optimization Controller**
- **Location**: `src/optimization/controller.py`
- **Responsibilities**:
  - Manage optimization iterations
  - Track performance metrics across iterations
  - Determine convergence or stopping criteria
  - Orchestrate forward and backward passes
  - Save optimization history and results

#### 2.2.2 **Forward Pass Evaluator**
- **Location**: `src/optimization/forward_pass.py`
- **Responsibilities**:
  - Load test dataset questions
  - Execute each question through the full agent system
  - Collect generated answers
  - Score answers using existing multi-critic system
  - Aggregate performance metrics per prompt/agent

#### 2.2.3 **Performance Analyzer**
- **Location**: `src/optimization/performance_analyzer.py`
- **Responsibilities**:
  - Analyze per-question performance
  - Identify patterns in failures vs successes
  - Group issues by prompt/agent type
  - Extract specific failure modes
  - Generate performance reports for debate agents

#### 2.2.4 **Multi-Agent Debate System for Prompt Review**
- **Location**: `src/optimization/debate/`
- **Specialized Agents**:
  
  a. **Performance Reviewer Agents** (Round 1: Independent Analysis)
     - `PerformanceReviewerAgent` - Analyzes test results for specific prompt
     - Different perspectives: "coverage_issues", "clarity_problems", "instruction_adherence"
     - Output: Identification of prompt weaknesses with evidence from test failures
  
  b. **Modification Proposer Agents** (Round 2: Adversarial Critique)
     - `ModificationProposerAgent` - Suggests specific prompt changes
     - Critiques others' proposals for effectiveness and feasibility
     - Output: Specific, actionable prompt modification proposals
  
  c. **Synthesis Agent** (Round 3: Cooperative Synthesis)
     - `PromptModificationSynthesisAgent` - Creates final coherent modifications
     - Resolves conflicts between proposals
     - Ensures modifications don't break existing functionality
     - Output: Final prompt modification plan with diffs

#### 2.2.5 **Prompt Modification Applier**
- **Location**: `src/optimization/backward_pass.py`
- **Responsibilities**:
  - Parse modification proposals
  - Apply changes to YAML prompt files
  - Validate modified prompts (syntax, variable consistency)
  - Create backup/versioning
  - Reload modified prompts into system

#### 2.2.6 **Improvement Tracker**
- **Location**: `src/optimization/metrics.py`
- **Responsibilities**:
  - Track performance metrics across iterations
  - Compare iteration N vs iteration N-1
  - Detect improvement, degradation, or stagnation
  - Generate improvement reports
  - Suggest early stopping

---

## 3. Data Models

### 3.1 Core Data Structures

```python
@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    test_dataset_path: str
    target_prompts: List[str]  # Which prompts to optimize
    max_iterations: int = 5
    performance_threshold: float = 0.85  # Stop if achieved
    improvement_threshold: float = 0.02  # Min improvement to continue
    debate_rounds: int = 3
    critics_per_round: int = 3
    backup_prompts: bool = True
    enable_validation: bool = True

@dataclass
class TestQuestion:
    """Test question with expected performance criteria."""
    id: str
    prompt: str
    category: str
    expected_key_points: List[str]
    target_agents: List[str]  # Which agents should handle this
    rubric_focus: List[str]
    difficulty: str

@dataclass
class QuestionResult:
    """Result of executing a single test question."""
    question_id: str
    generated_answer: str
    critic_scores: Dict[str, CriticScore]  # critic_type → score
    overall_score: float
    execution_time_ms: float
    agent_chain: List[str]  # Which agents were involved
    prompt_versions: Dict[str, str]  # agent → prompt hash
    
@dataclass
class PromptPerformance:
    """Performance metrics for a specific prompt."""
    prompt_name: str
    prompt_file: str
    questions_evaluated: int
    average_score: float
    score_by_dimension: Dict[str, float]
    failure_patterns: List[str]
    success_patterns: List[str]
    low_scoring_questions: List[str]  # Question IDs
    
@dataclass
class PromptModificationProposal:
    """Proposed modification to a prompt."""
    prompt_name: str
    modification_type: str  # "add_instruction", "clarify_section", "add_example", etc.
    target_section: str  # Which part of prompt
    original_text: str
    proposed_text: str
    rationale: str
    expected_improvement: str
    confidence: float
    supporting_evidence: List[str]  # Question IDs that motivate this
    
@dataclass
class IterationResult:
    """Results from one optimization iteration."""
    iteration_number: int
    timestamp: str
    test_results: List[QuestionResult]
    prompt_performances: Dict[str, PromptPerformance]
    overall_performance: float
    modification_proposals: List[PromptModificationProposal]
    modifications_applied: List[PromptModificationProposal]
    improvement_over_previous: Optional[float]
    
@dataclass
class OptimizationRun:
    """Complete optimization run with all iterations."""
    run_id: str
    config: OptimizationConfig
    iterations: List[IterationResult]
    final_performance: float
    total_improvement: float
    convergence_achieved: bool
    stopping_reason: str
    total_runtime_minutes: float
```

---

## 4. Implementation Checklist

### Phase 1: Foundation & Data Management ✓ (Prerequisites Met)
- [x] Review existing architecture (completed in deep research)
- [x] Validate test dataset structure (exists: `configs/calibration_questions.json`)
- [x] Confirm prompt template system (exists: `src/agents/templates.py`)
- [x] Verify multi-critic evaluation works (exists: `src/critics/`)

### Phase 2: Core Infrastructure (Week 1-2)

#### 2.1 Data Models & Configuration
- [ ] Create `src/optimization/__init__.py`
- [ ] Define data models in `src/optimization/models.py`
  - [ ] OptimizationConfig
  - [ ] TestQuestion (extend existing Question model)
  - [ ] QuestionResult
  - [ ] PromptPerformance
  - [ ] PromptModificationProposal
  - [ ] IterationResult
  - [ ] OptimizationRun
- [ ] Create configuration schema in `configs/optimization_config.yaml`
- [ ] Implement configuration loader with validation

#### 2.2 Test Dataset Enhancement
- [ ] Extend `src/scoring/dataset.py` with optimization-specific methods
  - [ ] `get_target_agent_mapping()` - map questions to agents
  - [ ] `validate_coverage()` - ensure all prompts tested
  - [ ] `create_optimization_subset()` - select representative questions
- [ ] Create test dataset validator
- [ ] Add question → prompt mapping logic

#### 2.3 Metrics & Tracking System
- [ ] Create `src/optimization/metrics.py`
  - [ ] `PerformanceMetrics` class
  - [ ] `ImprovementCalculator` class  
  - [ ] `ConvergenceDetector` class
- [ ] Implement metric aggregation functions
  - [ ] Per-question metrics
  - [ ] Per-prompt metrics
  - [ ] Per-dimension metrics
  - [ ] Iteration-over-iteration delta calculation
- [ ] Create metrics visualization utilities (JSON export for analysis)

### Phase 3: Forward Pass Implementation (Week 2-3)

#### 3.1 Test Execution Engine
- [ ] Create `src/optimization/forward_pass.py`
- [ ] Implement `TestExecutor` class
  - [ ] Load test dataset
  - [ ] Execute questions through agent system
  - [ ] Collect answers and intermediate results
  - [ ] Handle errors gracefully (record failures)
- [ ] Integrate with existing agent system
  - [ ] District, School, Teacher agents
  - [ ] Track which prompts used for each question
- [ ] Implement batch execution with progress tracking

#### 3.2 Automated Scoring Integration
- [ ] Create `src/optimization/scoring.py`
- [ ] Implement `AutomatedScorer` class
  - [ ] Use existing multi-critic orchestrator
  - [ ] Score each generated answer
  - [ ] Collect dimension-wise scores
  - [ ] Aggregate to overall performance
- [ ] Add scoring validation
  - [ ] Verify all dimensions scored
  - [ ] Check for score consistency
- [ ] Create scoring report generator

#### 3.3 Performance Analysis
- [ ] Create `src/optimization/performance_analyzer.py`
- [ ] Implement `PromptPerformanceAnalyzer` class
  - [ ] Group results by prompt
  - [ ] Identify failure patterns
  - [ ] Extract success patterns
  - [ ] Generate prompt-specific performance reports
- [ ] Implement statistical analysis
  - [ ] Score distributions per prompt
  - [ ] Correlation between prompt features and performance
  - [ ] Identify outlier questions
- [ ] Create evidence extraction for debate
  - [ ] Low-scoring examples with explanations
  - [ ] High-scoring examples for pattern recognition

### Phase 4: Multi-Agent Debate for Prompt Review (Week 3-4)

#### 4.1 Debate Agent Prompts
- [ ] Create `configs/prompts/optimization/` directory
- [ ] Design `performance_reviewer.yaml`
  - [ ] Analyzes test results for specific prompt
  - [ ] Identifies weaknesses with evidence
  - [ ] Different perspectives (coverage, clarity, etc.)
- [ ] Design `modification_proposer.yaml`
  - [ ] Proposes specific prompt changes
  - [ ] Provides rationale and expected impact
  - [ ] Critiques other proposals
- [ ] Design `prompt_synthesis.yaml`
  - [ ] Synthesizes multiple proposals
  - [ ] Resolves conflicts
  - [ ] Creates coherent modification plan

#### 4.2 Performance Review Agents (Round 1)
- [ ] Create `src/optimization/debate/performance_reviewer.py`
- [ ] Implement `PerformanceReviewerAgent` class
  - [ ] Extends BaseAgent
  - [ ] Analyzes PromptPerformance data
  - [ ] Outputs weakness identification
- [ ] Create specialized reviewer agents
  - [ ] `CoverageReviewer` - focuses on completeness issues
  - [ ] `ClarityReviewer` - focuses on instruction clarity
  - [ ] `ConsistencyReviewer` - focuses on output consistency
- [ ] Implement parallel execution of reviewers

#### 4.3 Modification Proposer Agents (Round 2)
- [ ] Create `src/optimization/debate/modification_proposer.py`
- [ ] Implement `ModificationProposerAgent` class
  - [ ] Takes performance review as input
  - [ ] Generates PromptModificationProposal objects
  - [ ] Critiques other proposals
- [ ] Implement proposal validation
  - [ ] Ensure proposals are specific and actionable
  - [ ] Check for conflicts with existing prompt structure
  - [ ] Validate variable consistency
- [ ] Create adversarial critique mechanism
  - [ ] Agents review each other's proposals
  - [ ] Identify potential issues
  - [ ] Refine proposals based on critique

#### 4.4 Synthesis Agent (Round 3)
- [ ] Create `src/optimization/debate/synthesis_agent.py`
- [ ] Implement `PromptModificationSynthesisAgent` class
  - [ ] Aggregates all proposals for a prompt
  - [ ] Resolves conflicts between proposals
  - [ ] Creates coherent modification plan
  - [ ] Generates final diff-style changes
- [ ] Implement conflict resolution logic
  - [ ] Identify overlapping modifications
  - [ ] Prioritize based on confidence and evidence
  - [ ] Ensure consistency across changes
- [ ] Create validation for synthesized modifications
  - [ ] Check prompt syntax
  - [ ] Verify variable placeholders unchanged
  - [ ] Ensure modifications are reversible

#### 4.5 Debate Orchestrator
- [ ] Create `src/optimization/debate/orchestrator.py`
- [ ] Implement `PromptOptimizationDebateOrchestrator` class
  - [ ] Manages 3-round debate process
  - [ ] Coordinates reviewer → proposer → synthesis flow
  - [ ] Handles parallel execution within rounds
  - [ ] Tracks debate history
- [ ] Adapt existing multi-critic orchestrator pattern
- [ ] Add debate-specific configuration
  - [ ] Number of agents per round
  - [ ] Temperature/creativity settings
  - [ ] Timeout configurations

### Phase 5: Backward Pass Implementation (Week 4-5)

#### 5.1 Prompt Modification System
- [ ] Create `src/optimization/backward_pass.py`
- [ ] Implement `PromptModifier` class
  - [ ] Parses PromptModificationProposal objects
  - [ ] Loads current prompt YAML files
  - [ ] Applies modifications to prompts
  - [ ] Writes updated YAML files
- [ ] Implement diff-based modification
  - [ ] Generate human-readable diffs
  - [ ] Apply changes safely
  - [ ] Validate after each change
- [ ] Create prompt backup system
  - [ ] Version control for prompts
  - [ ] Rollback capability
  - [ ] Store modification history

#### 5.2 Validation & Safety
- [ ] Create `src/optimization/validators.py`
- [ ] Implement prompt validation
  - [ ] YAML syntax validation
  - [ ] Variable placeholder consistency
  - [ ] Template rendering validation
  - [ ] Schema compliance checking
- [ ] Implement safety checks
  - [ ] Detect breaking changes
  - [ ] Verify prompt still loads correctly
  - [ ] Test basic rendering with sample variables
- [ ] Create validation reports
  - [ ] List of changes applied
  - [ ] Validation results
  - [ ] Warnings and errors

#### 5.3 Prompt Reload System
- [ ] Implement dynamic prompt reloading
  - [ ] Clear template cache
  - [ ] Reload modified prompts
  - [ ] Verify agents pick up new prompts
- [ ] Add hot-reload capability for iteration speed
- [ ] Create prompt version tracking
  - [ ] Hash-based versioning
  - [ ] Prompt genealogy tracking

### Phase 6: Optimization Controller (Week 5-6)

#### 6.1 Main Controller Implementation
- [ ] Create `src/optimization/controller.py`
- [ ] Implement `PromptOptimizationController` class
  - [ ] Initialize optimization run
  - [ ] Manage iteration loop
  - [ ] Coordinate forward → debate → backward flow
  - [ ] Track iteration history
  - [ ] Handle errors and recovery
- [ ] Implement iteration logic
  - [ ] Execute forward pass
  - [ ] Run performance analysis
  - [ ] Trigger multi-agent debate
  - [ ] Apply modifications (backward pass)
  - [ ] Re-evaluate performance
  - [ ] Check convergence

#### 6.2 Convergence & Stopping Criteria
- [ ] Implement convergence detection
  - [ ] Performance threshold reached
  - [ ] Improvement below threshold for N iterations
  - [ ] Maximum iterations reached
  - [ ] Performance degradation detected
- [ ] Add early stopping logic
  - [ ] Detect overfitting to test set
  - [ ] Detect oscillation in performance
  - [ ] User-defined stopping conditions
- [ ] Implement rollback mechanism
  - [ ] Restore best-performing iteration
  - [ ] Handle failed modifications

#### 6.3 Persistence & Reporting
- [ ] Create `src/optimization/persistence.py`
- [ ] Implement optimization state saving
  - [ ] Save after each iteration
  - [ ] Resume from checkpoint
  - [ ] Export final results
- [ ] Create comprehensive reporting
  - [ ] Iteration-by-iteration performance
  - [ ] Modification history
  - [ ] Improvement graphs (data for plotting)
  - [ ] Final summary report
- [ ] Export formats
  - [ ] JSON for programmatic access
  - [ ] Markdown for human review
  - [ ] CSV for spreadsheet analysis

### Phase 7: CLI & Integration (Week 6)

#### 7.1 Command-Line Interface
- [ ] Create `scripts/run_prompt_optimization.py`
- [ ] Implement CLI with argparse
  - [ ] Configuration file input
  - [ ] Test dataset selection
  - [ ] Target prompts specification
  - [ ] Iteration controls
  - [ ] Output directory
- [ ] Add interactive mode
  - [ ] Show progress during optimization
  - [ ] Prompt for review before applying changes
  - [ ] Manual approval of modifications
- [ ] Create dry-run mode
  - [ ] Run debate without applying changes
  - [ ] Preview modifications

#### 7.2 Integration with Existing Systems
- [ ] Integrate with calibration system
  - [ ] Use calibration questions as test set
  - [ ] Compare human vs optimized performance
- [ ] Integrate with evaluation tracking
  - [ ] Store optimization runs in database
  - [ ] Link to experiment tracking
- [ ] Add to main orchestrator workflow
  - [ ] Optional optimization step
  - [ ] Pre-deployment optimization

#### 7.3 Monitoring & Observability
- [ ] Add logging throughout system
  - [ ] Structured logging with context
  - [ ] Different log levels for debug
  - [ ] Performance timing logs
- [ ] Create progress monitoring
  - [ ] Real-time iteration updates
  - [ ] Time estimates
  - [ ] Performance trend display
- [ ] Add health checks
  - [ ] LLM API availability
  - [ ] File system permissions
  - [ ] Configuration validation

### Phase 8: Testing & Validation (Week 7)

#### 8.1 Unit Tests
- [ ] Create `tests/test_optimization/`
- [ ] Test data models
  - [ ] `test_models.py` - data structure validation
- [ ] Test forward pass
  - [ ] `test_forward_pass.py` - execution and scoring
- [ ] Test performance analysis
  - [ ] `test_performance_analyzer.py` - metric calculation
- [ ] Test debate system
  - [ ] `test_debate_orchestrator.py` - multi-agent coordination
- [ ] Test backward pass
  - [ ] `test_prompt_modifier.py` - modification application
- [ ] Test controller
  - [ ] `test_controller.py` - iteration logic

#### 8.2 Integration Tests
- [ ] Create `tests/integration/test_optimization_flow.py`
- [ ] Test complete optimization cycle
  - [ ] End-to-end with mock LLM
  - [ ] Forward → debate → backward flow
  - [ ] Multiple iterations
- [ ] Test with real test dataset
  - [ ] Use subset of calibration questions
  - [ ] Verify actual improvement
- [ ] Test error handling
  - [ ] Failed modifications
  - [ ] Invalid proposals
  - [ ] LLM API failures

#### 8.3 Validation Suite
- [ ] Create validation test suite
  - [ ] Prompt modification safety
  - [ ] Backwards compatibility
  - [ ] Performance regression detection
- [ ] Benchmark optimization effectiveness
  - [ ] Measure typical improvement rates
  - [ ] Determine optimal iteration counts
  - [ ] Validate stopping criteria

### Phase 9: Documentation (Week 7-8)

#### 9.1 Technical Documentation
- [ ] Create `docs/optimization/` directory
- [ ] Write architecture documentation
  - [ ] System overview
  - [ ] Component interactions
  - [ ] Data flow diagrams
- [ ] Write API documentation
  - [ ] All public classes and methods
  - [ ] Configuration options
  - [ ] Extension points
- [ ] Create developer guide
  - [ ] How to add new debate agents
  - [ ] Custom modification validators
  - [ ] Custom metrics

#### 9.2 User Documentation
- [ ] Write user guide
  - [ ] Quick start tutorial
  - [ ] Configuration guide
  - [ ] Best practices
- [ ] Create examples
  - [ ] Basic optimization run
  - [ ] Advanced configuration
  - [ ] Custom test datasets
- [ ] Write troubleshooting guide
  - [ ] Common issues
  - [ ] Error messages
  - [ ] Performance tuning

#### 9.3 Research Documentation
- [ ] Document optimization methodology
  - [ ] Theoretical foundation
  - [ ] Multi-agent debate rationale
  - [ ] Comparison to alternatives
- [ ] Create evaluation report template
  - [ ] How to assess optimization quality
  - [ ] Metrics interpretation
  - [ ] Statistical significance

### Phase 10: Production Readiness (Week 8)

#### 10.1 Performance Optimization
- [ ] Profile system performance
  - [ ] Identify bottlenecks
  - [ ] Optimize slow operations
- [ ] Implement caching
  - [ ] Prompt template caching
  - [ ] Result caching for repeatability
- [ ] Add parallelization
  - [ ] Parallel question execution
  - [ ] Parallel debate agents

#### 10.2 Error Handling & Resilience
- [ ] Comprehensive error handling
  - [ ] Graceful degradation
  - [ ] Retry logic for LLM calls
  - [ ] Checkpoint recovery
- [ ] Add validation at all boundaries
  - [ ] Input validation
  - [ ] Output validation
  - [ ] State validation
- [ ] Create error recovery procedures
  - [ ] Rollback failed modifications
  - [ ] Resume from last good state

#### 10.3 Deployment & CI/CD
- [ ] Add to CI/CD pipeline
  - [ ] Run optimization tests
  - [ ] Validate prompt modifications don't break system
- [ ] Create deployment checklist
  - [ ] Backup current prompts
  - [ ] Run validation suite
  - [ ] Gradual rollout strategy
- [ ] Set up monitoring
  - [ ] Track optimization runs
  - [ ] Alert on failures
  - [ ] Performance dashboards

---

## 5. Key Technical Decisions

### 5.1 Test Dataset Size
- **Recommendation**: 20-50 questions per optimization run
- **Rationale**: Balance between coverage and execution time
- **Tradeoff**: Larger = better optimization signal, but slower iterations

### 5.2 Debate Configuration
- **Rounds**: 3 (Independent → Adversarial → Synthesis)
- **Agents per round**: 3 reviewers, 3 proposers, 1 synthesizer
- **Model selection**: 
  - Round 1: Haiku (fast, diverse initial analysis)
  - Round 2-3: Sonnet (deep reasoning for proposals)

### 5.3 Modification Granularity
- **Target level**: Section-level modifications (not word-level)
- **Rationale**: Maintain prompt coherence, easier to review
- **Examples**: Add instruction section, clarify role description, add examples

### 5.4 Iteration Strategy
- **Max iterations**: 5 (default)
- **Early stopping**: If improvement < 2% for 2 consecutive iterations
- **Performance threshold**: 85% average score across test set

### 5.5 Safety & Validation
- **Backup all prompts** before each modification
- **Validate** after each change (syntax, variables, rendering)
- **Manual approval mode** available for production systems
- **Rollback** to best iteration if final performance degrades

---

## 6. Risks & Mitigations

### 6.1 Risk: Overfitting to Test Set
- **Mitigation**: Use diverse test set, hold out validation set, limit iterations
- **Detection**: Monitor performance on unseen questions periodically

### 6.2 Risk: Breaking Existing Functionality
- **Mitigation**: Comprehensive validation, backup system, gradual rollout
- **Detection**: Run existing test suite after modifications

### 6.3 Risk: Unstable Modifications
- **Mitigation**: Synthesizer agent ensures coherence, human review option
- **Detection**: Performance degradation triggers rollback

### 6.4 Risk: High LLM API Costs
- **Mitigation**: Batch operations, use cheaper models where possible, caching
- **Estimation**: ~50 questions × 5 iterations × 10 LLM calls = 2500 API calls
  - With debate: +7 agents × 5 iterations × 14 prompts = +490 calls
  - Total: ~3000 calls per optimization run

### 6.5 Risk: Long Execution Time
- **Mitigation**: Parallel execution, incremental optimization, resume capability
- **Estimation**: ~2-4 hours for full optimization run (5 iterations, 50 questions)

---

## 7. Success Metrics

### 7.1 System Performance Metrics
- **Primary**: Average score improvement across test set (target: +10-15%)
- **Secondary**: Improvement in specific dimensions (coverage, depth, etc.)
- **Tertiary**: Reduction in score variance (more consistent performance)

### 7.2 Operational Metrics
- **Iteration time**: <30 minutes per iteration
- **Modification success rate**: >90% of proposals applied without errors
- **Convergence rate**: Reach threshold within 5 iterations for 80% of runs

### 7.3 Quality Metrics
- **Human evaluation alignment**: Improved prompts score higher with human evaluators
- **Generalization**: Performance improvement on hold-out set >5%
- **Stability**: No degradation on existing test suites

---

## 8. Future Enhancements (Post-MVP)

### 8.1 Advanced Optimization Techniques
- [ ] Multi-objective optimization (balance multiple dimensions)
- [ ] Prompt feature extraction and analysis
- [ ] Transfer learning from similar prompt domains
- [ ] Genetic algorithm for prompt variation

### 8.2 Expanded Debate System
- [ ] More specialized reviewer perspectives
- [ ] Dynamic agent assignment based on failure types
- [ ] Meta-learning: agents learn from past optimization runs

### 8.3 Automation & Integration
- [ ] Continuous optimization on production data
- [ ] A/B testing framework for prompt variants
- [ ] Automatic prompt versioning and deployment
- [ ] Integration with prompt registry/library

### 8.4 Observability & Analysis
- [ ] Visualization dashboard for optimization runs
- [ ] Prompt diff visualization tool
- [ ] Performance correlation analysis
- [ ] Automatic insight generation from optimization history

---

## 9. Dependencies

### 9.1 Existing Codebase Components
- `src/agents/base.py` - BaseAgent class
- `src/agents/templates.py` - Template management
- `src/critics/orchestrator.py` - Multi-agent orchestration pattern
- `src/critics/multi_critic.py` - Critic agents
- `src/scoring/dataset.py` - Test dataset management
- `src/eval/calibration.py` - Calibration and alignment metrics
- `src/utils/llm.py` - LLM client
- `configs/prompts/*.yaml` - Existing prompt templates

### 9.2 External Libraries
- Standard library: `asyncio`, `logging`, `pathlib`, `dataclasses`
- Data processing: `numpy`, `scipy` (already present)
- YAML parsing: `pyyaml` (already present)
- Testing: `pytest` (already present)
- Optional: `rich` for CLI progress displays

### 9.3 Infrastructure
- LLM API access (Claude Sonnet 4.5, Haiku)
- File system access for prompt storage
- Sufficient disk space for iteration history
- Compute for parallel agent execution

---

## 10. Timeline Estimate

**Total Duration**: 7-8 weeks for full implementation and testing

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| **Phase 2**: Core Infrastructure | 1-2 weeks | None | Data models, config, metrics |
| **Phase 3**: Forward Pass | 1 week | Phase 2 | Test execution, scoring, analysis |
| **Phase 4**: Multi-Agent Debate | 1-2 weeks | Phase 2, 3 | Debate agents, orchestrator |
| **Phase 5**: Backward Pass | 1 week | Phase 2 | Prompt modification, validation |
| **Phase 6**: Controller | 1 week | Phase 2-5 | Optimization loop, convergence |
| **Phase 7**: CLI & Integration | 1 week | Phase 6 | Command-line tool, integration |
| **Phase 8**: Testing | 1 week | Phase 2-7 | Test suites, validation |
| **Phase 9**: Documentation | Parallel | All phases | Docs, guides, examples |
| **Phase 10**: Production Ready | 1 week | Phase 2-8 | Performance, resilience |

**Milestones:**
- **Week 2**: Forward pass working, can execute test dataset
- **Week 4**: Multi-agent debate generates modification proposals
- **Week 5**: Complete iteration cycle working (forward → debate → backward)
- **Week 6**: Full optimization controller with convergence
- **Week 7**: Tested and validated on real prompts
- **Week 8**: Production-ready with documentation

---

## 11. Getting Started

### 11.1 Prerequisites
1. Review this implementation plan
2. Ensure test dataset is prepared (`configs/calibration_questions.json`)
3. Verify existing agent system is working
4. Set up development environment

### 11.2 First Steps
1. Create directory structure:
   ```bash
   mkdir -p src/optimization/debate
   mkdir -p configs/prompts/optimization
   mkdir -p tests/test_optimization
   mkdir -p docs/optimization
   ```

2. Implement Phase 2.1 (Data Models):
   - Start with `src/optimization/models.py`
   - Define all core data structures
   - Add validation and serialization

3. Set up testing infrastructure:
   - Create test fixtures for sample data
   - Set up mock LLM for testing
   - Create baseline test suite

### 11.3 Development Workflow
1. Work in feature branches
2. Write tests before implementation (TDD)
3. Review each phase with stakeholders
4. Validate on small test set before full runs
5. Document as you build

---

## 12. References & Related Work

### 12.1 Existing Codebase References
- Multi-agent debate: `.claude/commands/debate-solve.md`
- Multi-critic system: `src/critics/orchestrator.py`
- Calibration: `src/eval/calibration.py`
- Test questions: `configs/calibration_questions.json`

### 12.2 Relevant Concepts
- **Prompt optimization**: Automated improvement of LLM prompts
- **Multi-agent debate (MAD)**: Using multiple agents with different perspectives
- **Backpropagation analogy**: Iterative refinement based on performance feedback
- **Meta-learning**: System learns how to improve itself

### 12.3 Similar Systems
- DSPy (prompt optimization through examples)
- PromptBreeder (evolutionary prompt optimization)
- OPRO (optimization by prompting)
- AutoPrompt (gradient-based prompt search)

**Key Difference**: This system uses multi-agent reasoning and debate rather than gradient descent or evolutionary algorithms, making it more interpretable and providing natural language rationale for changes.

---

## Appendix A: Example Workflow

### A.1 Complete Optimization Run Example

```
1. Initialize Optimization
   - Load config: configs/optimization_config.yaml
   - Load test dataset: configs/calibration_questions.json (50 questions)
   - Target prompts: [teacher_agent.yaml, coverage_critic.yaml]
   - Max iterations: 5

2. Iteration 1 - Baseline
   Forward Pass:
   - Execute 50 questions through system
   - Scores: avg=72.3, coverage=68, depth=75, style=74
   
   Performance Analysis:
   - teacher_agent: avg=71, failures on 12 questions
   - coverage_critic: avg=68, weak on complex questions
   
   Multi-Agent Debate:
   Round 1 (Reviewers):
   - CoverageReviewer: "teacher_agent lacks explicit instruction to cover all key points"
   - ClarityReviewer: "coverage_critic's 'breadth' criterion unclear"
   
   Round 2 (Proposers):
   - Proposer 1: "Add checklist instruction to teacher_agent"
   - Proposer 2: "Define 'breadth' with examples in coverage_critic"
   
   Round 3 (Synthesis):
   - Final mods: 2 modifications proposed
   
   Backward Pass:
   - Apply modifications to YAML files
   - Validate: ✓ All valid
   - Reload prompts

3. Iteration 2 - First Improvement
   Forward Pass:
   - Execute 50 questions with modified prompts
   - Scores: avg=76.8, coverage=74, depth=76, style=75
   - Improvement: +4.5 points (+6.2%)
   
   [Process repeats...]

4. Iteration 5 - Convergence
   Forward Pass:
   - Scores: avg=84.2, coverage=83, depth=85, style=84
   - Improvement over iteration 4: +1.1 points (+1.3%)
   
   Convergence Check:
   - Improvement <2% threshold
   - Overall score <85% threshold
   - Decision: Continue one more iteration OR stop

5. Final Results
   - Initial performance: 72.3
   - Final performance: 84.2
   - Total improvement: +11.9 points (+16.5%)
   - Iterations: 5
   - Modifications applied: 8
   - Runtime: 2.5 hours
```

### A.2 Example Modification Proposal

```yaml
prompt_modification_proposal:
  prompt_name: "teacher_agent"
  modification_type: "add_instruction"
  target_section: "task_description"
  
  original_text: |
    You are a teacher agent responsible for generating 
    comprehensive answers to questions.
  
  proposed_text: |
    You are a teacher agent responsible for generating 
    comprehensive answers to questions.
    
    When answering, ensure you:
    1. Address all key points explicitly mentioned in the question
    2. Provide specific examples or evidence for each claim
    3. Organize your response with clear structure
    4. Use appropriate technical depth for the domain
  
  rationale: |
    Analysis of 12 failed test cases (IDs: hma-001, hma-004, sql-002, ...)
    revealed that the agent often omits key points from the question,
    especially when questions have multiple parts. Adding an explicit
    checklist instruction increased coverage score from 68 to 74 in
    initial testing.
  
  expected_improvement: "+5-7 points in coverage dimension"
  confidence: 0.82
  supporting_evidence:
    - "hma-001: Missed 'coordination mechanisms' key point"
    - "hma-004: Omitted 'load redistribution' discussion"
    - "sql-002: Failed to cover 'security considerations'"
```

---

## Appendix B: Configuration Example

```yaml
# configs/optimization_config.yaml

optimization:
  run_name: "baseline_optimization_v1"
  
  # Test dataset
  test_dataset:
    path: "configs/calibration_questions.json"
    question_limit: 50  # Use subset for faster iteration
    balance_categories: true
    random_seed: 42
  
  # Target prompts to optimize
  target_prompts:
    - "configs/prompts/teacher_agent.yaml"
    - "configs/prompts/coverage_critic.yaml"
    - "configs/prompts/depth_critic.yaml"
    # Can specify "all" to optimize all prompts
  
  # Iteration control
  iterations:
    max_iterations: 5
    performance_threshold: 0.85  # Stop if avg score ≥85%
    improvement_threshold: 0.02  # Stop if improvement <2% for 2 iterations
    min_improvement_iterations: 2
  
  # Debate configuration
  debate:
    enable: true
    rounds: 3
    reviewers_per_prompt: 3
    proposers_per_prompt: 3
    model:
      round_1: "claude-3-5-haiku"
      round_2: "claude-sonnet-4-5"
      round_3: "claude-sonnet-4-5"
  
  # Modification settings
  modification:
    require_approval: false  # true for manual review
    backup_prompts: true
    validation_strict: true
    max_modifications_per_iteration: 10
  
  # Scoring configuration
  scoring:
    use_multi_critic: true
    critic_timeout_ms: 30000
    enable_debate: true  # For scoring
  
  # Output and persistence
  output:
    results_dir: "optimization_results/"
    save_iterations: true
    export_format: ["json", "markdown"]
    verbose_logging: true
  
  # Safety and limits
  safety:
    max_prompt_length: 5000  # characters
    rollback_on_degradation: true
    preserve_variable_names: true
    validate_before_apply: true
```

---

## Appendix C: Prompt Template for Debate Agents

### C.1 Performance Reviewer Template

```yaml
# configs/prompts/optimization/performance_reviewer.yaml

name: performance_reviewer
description: Analyzes test execution results to identify prompt weaknesses
version: "1.0"
tags: ["optimization", "review", "performance-analysis"]

templates:
  - name: performance_review
    description: "Review prompt performance on test dataset"
    template: |
      You are a Performance Reviewer Agent analyzing the effectiveness of a specific prompt template.
      
      Your role: Review test execution results and identify weaknesses, gaps, and failure patterns in the prompt.
      
      ## Prompt Being Reviewed
      **Prompt Name**: $prompt_name
      **Prompt File**: $prompt_file
      
      **Current Prompt Content**:
      ```
      $current_prompt
      ```
      
      ## Performance Data
      
      **Overall Statistics**:
      - Questions Evaluated: $questions_evaluated
      - Average Score: $average_score / 100
      - Score by Dimension:
      $score_by_dimension
      
      **Low-Scoring Questions** (score < 70):
      $low_scoring_questions
      
      **Failure Patterns Detected**:
      $failure_patterns
      
      **Success Patterns**:
      $success_patterns
      
      ## Your Task
      
      Analyze this performance data from the perspective of **$perspective** 
      (e.g., "coverage completeness", "instruction clarity", "output consistency").
      
      Provide your analysis in the following format:
      
      {
        "reviewer_perspective": "$perspective",
        "identified_weaknesses": [
          {
            "weakness": "Clear description of the weakness",
            "evidence": ["Question IDs or patterns supporting this"],
            "severity": "high|medium|low",
            "affected_dimension": "coverage|depth|style|instruction_following"
          }
        ],
        "root_causes": [
          "Specific aspects of the prompt causing issues"
        ],
        "impact_assessment": {
          "questions_affected": 12,
          "average_score_impact": -8.5,
          "dimensions_affected": ["coverage", "depth"]
        },
        "confidence": 0.0-1.0
      }
      
      Focus on:
      1. Specific, actionable weaknesses (not vague observations)
      2. Evidence-based analysis (cite question IDs)
      3. Root cause identification (what in the prompt causes this?)
      4. Severity and impact quantification
      
    variables:
      - name: prompt_name
        description: "Name of the prompt being reviewed"
        required: true
        variable_type: str
      
      - name: prompt_file
        description: "File path of the prompt"
        required: true
        variable_type: str
      
      - name: current_prompt
        description: "Full text of the current prompt"
        required: true
        variable_type: str
      
      - name: questions_evaluated
        description: "Number of test questions"
        required: true
        variable_type: int
      
      - name: average_score
        description: "Average performance score"
        required: true
        variable_type: float
      
      - name: score_by_dimension
        description: "Dimension-wise scores breakdown"
        required: true
        variable_type: str
      
      - name: low_scoring_questions
        description: "Details of poorly performing questions"
        required: true
        variable_type: str
      
      - name: failure_patterns
        description: "Detected patterns in failures"
        required: true
        variable_type: str
      
      - name: success_patterns
        description: "Detected patterns in successes"
        required: true
        variable_type: str
      
      - name: perspective
        description: "Specific review perspective to focus on"
        required: true
        variable_type: str
```

---

**END OF IMPLEMENTATION PLAN**

*This is a living document. Update as implementation progresses and new insights emerge.*
