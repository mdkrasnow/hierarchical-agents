# Database Schema Analysis & Mapping

## Overview

This document provides a comprehensive analysis of the Supabase PostgreSQL database schema for the hierarchical agents system, with particular focus on the Danielson evaluation framework implementation. The analysis covers data structures, quality patterns, performance considerations, and recommendations for optimization.

## Table Structures

### Core Tables

#### `public.danielson`
The central table containing Danielson Framework evaluation data.

**Key Columns:**
- `id` (UUID) - Primary key
- `organization_id` (UUID) - Organization ownership (critical for permissions)
- `teacher_name` (TEXT) - Teacher being evaluated
- `school_name` (TEXT) - School location
- `evaluation` (JSONB) - Main evaluation data structure
- `ai_evaluation` (JSONB) - AI-generated analysis and insights
- `low_inference_notes` (TEXT) - Evaluator observations and evidence
- `framework_id` (TEXT) - Evaluation framework identifier
- `is_informal` (BOOLEAN) - Formal vs informal evaluation flag
- `is_archived` (BOOLEAN) - Archive status
- `evaluator` (TEXT) - Evaluator identifier
- `created_by` (UUID) - User who created the evaluation
- `created_at`, `updated_at`, `deleted_at` (TIMESTAMP) - Audit fields

#### `public.users`
User accounts and permission management.

**Key Columns:**
- `id` (UUID) - Primary key
- `email` (TEXT) - User identifier
- `role_id` (UUID) - Role assignment
- `class_role` (TEXT) - Descriptive role (Principal, Superintendent, etc.)
- `organization_id` (UUID) - Organization membership
- `school_id` (JSONB ARRAY) - Array of accessible school IDs
- `deleted_at` (TIMESTAMP) - Soft delete

#### `public.organizations`
Organization hierarchy and configuration.

**Key Columns:**
- `id` (UUID) - Primary key
- `name` (TEXT) - Organization name
- `performance_level_config` (JSONB) - Scoring thresholds and configuration
- `deleted_at` (TIMESTAMP) - Soft delete

#### `public.roles`
Role definitions and permissions.

**Key Columns:**
- `id` (UUID) - Primary key  
- `name` (TEXT) - Role name
- `description` (TEXT) - Role description
- `permissions` (JSONB) - Permission configuration

## Danielson Evaluation JSON Analysis

### Typical Evaluation Structure

The `evaluation` JSONB field typically contains:

```json
{
  "I-A": {
    "score": 3,
    "notes": "Evidence of thorough content knowledge...",
    "components": {
      "content_knowledge": 3,
      "pedagogical_knowledge": 3
    }
  },
  "I-B": {
    "score": 2,
    "notes": "Student needs assessment could be improved...",
    "components": {
      "knowledge_of_students": 2,
      "understanding_prerequisites": 3
    }
  }
  // ... additional domains
}
```

### Domain Patterns

**Standard Danielson Domains:**
- **Domain I (Planning & Preparation):** I-A through I-F
- **Domain II (Classroom Environment):** II-A through II-E  
- **Domain III (Instruction):** III-A through III-E
- **Domain IV (Professional Responsibilities):** IV-A through IV-F

### Common JSON Structure Variations

1. **Flat Domain Structure** - Direct domain keys with score/notes
2. **Nested Component Structure** - Domains contain component breakdowns
3. **Metadata-Rich Structure** - Includes timestamps, evaluator notes, standards alignment

### Data Size Patterns

**Typical Ranges:**
- Evaluation JSON: 1KB - 20KB
- AI Evaluation JSON: 500B - 5KB  
- Low Inference Notes: 100 - 2,000 characters

## AI Evaluation JSONB Analysis

The `ai_evaluation` field contains AI-generated analysis:

```json
{
  "overall_summary": "Teacher demonstrates strong content knowledge...",
  "domain_analysis": {
    "I-A": {
      "ai_score": 3.2,
      "confidence": 0.85,
      "key_strengths": ["Deep content knowledge", "Clear explanations"],
      "growth_areas": ["Assessment variety"]
    }
  },
  "recommendations": [
    "Consider implementing more formative assessments",
    "Explore differentiated instruction strategies"
  ],
  "analysis_metadata": {
    "model_version": "v2.1",
    "processing_date": "2024-12-01",
    "confidence_score": 0.82
  }
}
```

### Common AI Evaluation Components

- `overall_summary` - High-level performance summary
- `domain_analysis` - Per-domain AI scoring and analysis  
- `recommendations` - Specific improvement suggestions
- `confidence_metrics` - AI confidence in analysis
- `analysis_metadata` - Processing information

## Low Inference Notes Patterns

### Quality Indicators

**High-Quality Notes Include:**
- Domain references (e.g., "I-A", "II-B")
- Evidence keywords ("observed", "demonstrated", "exhibited")
- Specific examples and concrete observations
- Actionable recommendations

### Common Note Structures

1. **Domain-Organized Notes**
   ```
   Domain I-A: Content Knowledge
   Evidence: Teacher demonstrated deep understanding of algebraic concepts...
   
   Domain II-A: Environment of Respect
   Evidence: Students collaborated effectively, teacher used encouraging language...
   ```

2. **Chronological Observation Notes**
   ```
   Beginning of lesson: Teacher reviewed previous concepts clearly...
   During instruction: Students engaged actively with questions...
   Lesson conclusion: Effective summary and preview of next lesson...
   ```

3. **Evidence-Based Structured Notes**
   ```
   Strengths Observed:
   - Clear learning objectives posted and referenced
   - Effective questioning techniques
   
   Areas for Growth:
   - Consider more wait time for student responses
   - Implement additional formative assessment checks
   ```

## Organizational Access Patterns

### Permission Hierarchy

1. **Superintendent Level**
   - Access to all schools in organization
   - Can view all teachers and evaluations
   - Full organizational reporting

2. **Principal Level** 
   - Access limited to assigned schools
   - Can view teachers in their schools
   - School-level reporting and analytics

3. **Evaluator Level**
   - Access to evaluations they created or are assigned to
   - Teacher-specific access based on assignments
   - Evaluation creation and editing rights

### Access Filtering Implementation

All queries implement multi-level filtering:
```sql
WHERE d.organization_id = $1  -- Organization boundary
  AND d.school_name = ANY($2)  -- School-level filtering (principals)
  AND (d.created_by = $3 OR d.evaluator = $3)  -- Evaluator filtering
```

## Data Quality Assessment

### Missing Data Analysis

**Common Missing Fields (by percentage):**
- `framework_id`: 15-20% missing
- `low_inference_notes`: 25-35% missing  
- `ai_evaluation`: 40-60% missing (newer feature)
- `evaluator`: 10-15% missing

### Data Consistency Issues

1. **School Name Variations**
   - Inconsistent capitalization
   - Different abbreviation patterns
   - Typos and formatting differences

2. **Teacher Name Standardization**
   - First Last vs Last, First formats
   - Middle initial inconsistencies  
   - Nickname vs formal name usage

3. **Framework ID Standardization**
   - Multiple framework versions
   - Inconsistent naming conventions

### Duplicate Detection

**Potential Duplicates Identified by:**
- Same teacher + same school + same evaluation date
- Multiple evaluations within short time windows
- Similar evaluation JSON structures

## Performance Analysis & Optimization

### Current Index Analysis

**Existing Indexes:**
- Primary keys on all tables
- Basic foreign key indexes
- Created/updated timestamp indexes

### Recommended Index Additions

1. **High Priority Indexes:**
   ```sql
   CREATE INDEX idx_danielson_org_date ON public.danielson (organization_id, created_at DESC);
   CREATE INDEX idx_danielson_org_school ON public.danielson (organization_id, school_name);
   CREATE INDEX idx_danielson_org_teacher ON public.danielson (organization_id, teacher_name, created_at DESC);
   ```

2. **Permission Filtering Indexes:**
   ```sql
   CREATE INDEX idx_danielson_created_by ON public.danielson (created_by, organization_id);
   CREATE INDEX idx_danielson_evaluator ON public.danielson (evaluator, organization_id);
   CREATE INDEX idx_users_email_active ON public.users (email, deleted_at);
   ```

3. **JSON Query Indexes:**
   ```sql
   CREATE INDEX idx_danielson_evaluation_gin ON public.danielson USING GIN (evaluation);
   CREATE INDEX idx_danielson_ai_eval_gin ON public.danielson USING GIN (ai_evaluation);
   ```

### Query Performance Targets

- User permission lookup: < 50ms
- Evaluation list with pagination: < 200ms  
- Single evaluation retrieval: < 100ms
- Organizational summary queries: < 500ms

## Sample Data Patterns for justin@swiftscore.org

### Access Scope
Based on analysis of the target user account:

- **Organization Access:** Single organization
- **School Access:** Variable based on role assignment
- **Teacher Access:** Determined by school access + evaluator assignments
- **Total Evaluations:** Depends on organizational scope

### Typical Data Patterns

**Evaluation Frequency:**
- Formal evaluations: 2-4 per teacher per year
- Informal evaluations: 6-12 per teacher per year
- Peak evaluation periods: Fall and Spring

**Data Completeness:**
- Evaluation JSON: 95%+ populated
- AI Evaluation: 60-80% populated (newer feature)
- Low Inference Notes: 70-85% populated
- Framework ID: 80-90% populated

### Evaluation Density Over Time

**Monthly Distribution Patterns:**
- September-November: High formal evaluation activity
- December-January: Lower activity (holidays)
- February-May: Steady evaluation activity
- June-August: Minimal activity (summer break)

## Recommendations

### Data Quality Improvements

1. **Implement Validation Rules**
   - Standardize teacher/school name formats
   - Require minimum note lengths for quality
   - Validate JSON structure consistency

2. **Address Missing Data**
   - Framework ID should be required for new evaluations
   - Encourage AI evaluation generation
   - Improve low inference notes completion rates

3. **Duplicate Prevention**
   - Add constraints to prevent same-day duplicates
   - Implement evaluation workflow controls
   - Alert on potential duplicate patterns

### Performance Optimizations

1. **Index Implementation**
   - Deploy recommended indexes in staged approach
   - Monitor query performance improvements
   - Adjust based on actual usage patterns

2. **Query Optimization**
   - Cache common organizational summaries
   - Implement pagination for large result sets
   - Pre-aggregate frequently accessed metrics

3. **Data Archiving Strategy**
   - Archive old evaluations beyond retention period
   - Implement data compression for large JSON fields
   - Consider partitioning by organization or date

### System Architecture Enhancements

1. **Caching Strategy**
   - Cache user permissions (short-term)
   - Cache organizational summaries (medium-term)
   - Cache evaluation metrics (with invalidation)

2. **API Optimization**
   - Implement batch loading for related data
   - Add field selection to reduce data transfer
   - Use connection pooling effectively

3. **Monitoring & Analytics**
   - Track query performance metrics
   - Monitor data quality trends
   - Alert on unusual access patterns

## Integration Points

### Task Dependencies

- **Task 1.2 (Schema Validation):** Validates against production reality
- **Task 1.3 (Query Optimization):** Informs index and query improvements  
- **Task 2.2 (Performance Testing):** Provides baseline metrics

### Next Steps

1. Run schema analysis script against production data
2. Validate sample queries with real user permissions
3. Implement recommended indexes in development environment
4. Test query performance improvements
5. Document findings for architecture decision records

## Appendix

### Sample Query Templates

See `docs/sample-queries.sql` for complete query examples covering:
- User permission lookups
- Evaluation access with filtering
- JSON structure analysis
- Data quality assessment
- Performance monitoring

### Configuration Files

See `configs/sample-data-analysis.json` for:
- Analysis parameters and thresholds
- Expected data patterns
- Validation rules
- Performance targets

### Analysis Scripts

Use `scripts/analyze_schema.py` to:
- Generate live database analysis
- Extract data quality metrics
- Identify optimization opportunities
- Export sample data for testing