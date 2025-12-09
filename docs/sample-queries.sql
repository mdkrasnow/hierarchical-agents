-- Sample Database Queries for Danielson Evaluation System
-- These queries demonstrate common access patterns and data exploration
-- for the hierarchical agents system with proper permission filtering

-- ============================================================================
-- USER AND PERMISSION QUERIES
-- ============================================================================

-- Get user permissions and organizational access for justin@swiftscore.org
SELECT 
    u.id as user_id,
    u.email,
    u.role_id,
    u.class_role,
    u.organization_id,
    u.school_id as school_ids,
    r.name as role_name,
    o.name as organization_name
FROM public.users u
LEFT JOIN public.roles r ON u.role_id = r.id
LEFT JOIN public.organizations o ON u.organization_id = o.id
WHERE u.email = 'justin@swiftscore.org' 
    AND u.deleted_at IS NULL;

-- Get all users in the same organization as justin@swiftscore.org
WITH target_user AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    u.email,
    u.class_role,
    r.name as role_name,
    u.school_id as accessible_schools,
    u.created_at
FROM public.users u
LEFT JOIN public.roles r ON u.role_id = r.id
CROSS JOIN target_user tu
WHERE u.organization_id = tu.organization_id 
    AND u.deleted_at IS NULL
ORDER BY u.class_role, u.email;

-- ============================================================================
-- DANIELSON EVALUATION QUERIES
-- ============================================================================

-- Get recent evaluations for justin@swiftscore.org's organization
WITH user_org AS (
    SELECT organization_id, school_id
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.id,
    d.teacher_name,
    d.school_name,
    d.framework_id,
    d.is_informal,
    d.is_archived,
    d.created_at,
    d.evaluator,
    LENGTH(d.evaluation::text) as eval_json_size,
    LENGTH(d.ai_evaluation::text) as ai_eval_size,
    LENGTH(d.low_inference_notes) as notes_length
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
ORDER BY d.created_at DESC
LIMIT 20;

-- Evaluation distribution by school and teacher
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.school_name,
    d.teacher_name,
    COUNT(*) as total_evaluations,
    COUNT(*) FILTER (WHERE d.is_informal = true) as informal_evaluations,
    COUNT(*) FILTER (WHERE d.is_archived = true) as archived_evaluations,
    MIN(d.created_at) as first_evaluation,
    MAX(d.created_at) as latest_evaluation,
    ARRAY_AGG(d.framework_id) FILTER (WHERE d.framework_id IS NOT NULL) as frameworks_used
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
GROUP BY d.school_name, d.teacher_name
ORDER BY total_evaluations DESC, d.school_name, d.teacher_name;

-- ============================================================================
-- JSON STRUCTURE ANALYSIS QUERIES
-- ============================================================================

-- Analyze evaluation JSON structure patterns
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.framework_id,
    jsonb_object_keys(d.evaluation) as eval_top_level_keys,
    COUNT(*) as frequency
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
    AND d.evaluation IS NOT NULL
GROUP BY d.framework_id, jsonb_object_keys(d.evaluation)
ORDER BY d.framework_id, frequency DESC;

-- Extract domain scores from evaluation JSON
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
),
domain_scores AS (
    SELECT 
        d.id,
        d.teacher_name,
        d.school_name,
        d.created_at,
        domain_key,
        domain_value,
        domain_value ->> 'score' as domain_score,
        domain_value ->> 'notes' as domain_notes
    FROM public.danielson d
    CROSS JOIN user_org uo,
    jsonb_each(d.evaluation) as domain_data(domain_key, domain_value)
    WHERE d.organization_id = uo.organization_id
        AND d.deleted_at IS NULL
        AND d.evaluation IS NOT NULL
        AND domain_key ~ '^[IVX]+[A-Z]?-[A-Z]$' -- Match domain patterns like "I-A", "II-B"
)
SELECT 
    domain_key,
    COUNT(*) as total_evaluations,
    COUNT(domain_score) as evaluations_with_scores,
    AVG(domain_score::numeric) FILTER (WHERE domain_score ~ '^[0-9]+\.?[0-9]*$') as avg_score,
    COUNT(domain_notes) FILTER (WHERE LENGTH(domain_notes) > 0) as evaluations_with_notes
FROM domain_scores
GROUP BY domain_key
ORDER BY domain_key;

-- AI evaluation analysis
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    jsonb_object_keys(d.ai_evaluation) as ai_eval_keys,
    COUNT(*) as frequency,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
    AND d.ai_evaluation IS NOT NULL
GROUP BY jsonb_object_keys(d.ai_evaluation)
ORDER BY frequency DESC;

-- ============================================================================
-- LOW INFERENCE NOTES ANALYSIS
-- ============================================================================

-- Notes quality analysis
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.school_name,
    COUNT(*) as total_evaluations,
    COUNT(d.low_inference_notes) FILTER (WHERE LENGTH(d.low_inference_notes) > 0) as evaluations_with_notes,
    ROUND(AVG(LENGTH(d.low_inference_notes)) FILTER (WHERE d.low_inference_notes IS NOT NULL), 0) as avg_notes_length,
    ROUND(AVG(array_length(string_to_array(d.low_inference_notes, ' '), 1)) FILTER (WHERE d.low_inference_notes IS NOT NULL), 0) as avg_word_count
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
GROUP BY d.school_name
ORDER BY total_evaluations DESC;

-- Find evaluations with domain references in notes
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.id,
    d.teacher_name,
    d.school_name,
    d.created_at,
    LENGTH(d.low_inference_notes) as notes_length,
    CASE 
        WHEN d.low_inference_notes ~* '\b[IVX]+[A-Z]?-[A-Z]\b' THEN 'Has Domain References'
        WHEN d.low_inference_notes ~* '(evidence|observed|demonstrated)' THEN 'Has Evidence Keywords'
        WHEN d.low_inference_notes ~* '(recommend|suggest|growth|improve)' THEN 'Has Recommendations'
        ELSE 'Basic Notes'
    END as notes_quality_category
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
    AND d.low_inference_notes IS NOT NULL
    AND LENGTH(d.low_inference_notes) > 20
ORDER BY d.created_at DESC
LIMIT 50;

-- ============================================================================
-- TEMPORAL AND DISTRIBUTION ANALYSIS
-- ============================================================================

-- Evaluation activity over time
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    DATE_TRUNC('month', d.created_at) as evaluation_month,
    COUNT(*) as total_evaluations,
    COUNT(DISTINCT d.teacher_name) as unique_teachers,
    COUNT(DISTINCT d.school_name) as unique_schools,
    COUNT(*) FILTER (WHERE d.is_informal = false) as formal_evaluations,
    COUNT(*) FILTER (WHERE d.ai_evaluation IS NOT NULL) as evaluations_with_ai
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
    AND d.created_at >= CURRENT_DATE - INTERVAL '24 months'
GROUP BY DATE_TRUNC('month', d.created_at)
ORDER BY evaluation_month DESC;

-- Teacher evaluation frequency analysis
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
),
teacher_stats AS (
    SELECT 
        d.teacher_name,
        d.school_name,
        COUNT(*) as total_evaluations,
        MIN(d.created_at) as first_eval,
        MAX(d.created_at) as latest_eval,
        MAX(d.created_at) - MIN(d.created_at) as evaluation_span
    FROM public.danielson d
    CROSS JOIN user_org uo
    WHERE d.organization_id = uo.organization_id
        AND d.deleted_at IS NULL
    GROUP BY d.teacher_name, d.school_name
)
SELECT 
    CASE 
        WHEN total_evaluations = 1 THEN 'Single Evaluation'
        WHEN total_evaluations BETWEEN 2 AND 5 THEN 'Low Frequency (2-5)'
        WHEN total_evaluations BETWEEN 6 AND 15 THEN 'Medium Frequency (6-15)'
        ELSE 'High Frequency (15+)'
    END as evaluation_frequency_category,
    COUNT(*) as teacher_count,
    ROUND(AVG(total_evaluations), 1) as avg_evaluations_per_teacher,
    ROUND(AVG(EXTRACT(days FROM evaluation_span)), 0) as avg_span_days
FROM teacher_stats
GROUP BY 
    CASE 
        WHEN total_evaluations = 1 THEN 'Single Evaluation'
        WHEN total_evaluations BETWEEN 2 AND 5 THEN 'Low Frequency (2-5)'
        WHEN total_evaluations BETWEEN 6 AND 15 THEN 'Medium Frequency (6-15)'
        ELSE 'High Frequency (15+)'
    END
ORDER BY 
    CASE 
        WHEN evaluation_frequency_category = 'Single Evaluation' THEN 1
        WHEN evaluation_frequency_category = 'Low Frequency (2-5)' THEN 2
        WHEN evaluation_frequency_category = 'Medium Frequency (6-15)' THEN 3
        ELSE 4
    END;

-- ============================================================================
-- DATA QUALITY AND COMPLETENESS QUERIES  
-- ============================================================================

-- Data completeness report
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    'Total Evaluations' as metric,
    COUNT(*) as count,
    100.0 as percentage
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id AND d.deleted_at IS NULL

UNION ALL

SELECT 
    'Has Evaluation JSON' as metric,
    COUNT(*) FILTER (WHERE d.evaluation IS NOT NULL) as count,
    ROUND(COUNT(*) FILTER (WHERE d.evaluation IS NOT NULL) * 100.0 / COUNT(*), 1) as percentage
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id AND d.deleted_at IS NULL

UNION ALL

SELECT 
    'Has AI Evaluation' as metric,
    COUNT(*) FILTER (WHERE d.ai_evaluation IS NOT NULL) as count,
    ROUND(COUNT(*) FILTER (WHERE d.ai_evaluation IS NOT NULL) * 100.0 / COUNT(*), 1) as percentage
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id AND d.deleted_at IS NULL

UNION ALL

SELECT 
    'Has Low Inference Notes' as metric,
    COUNT(*) FILTER (WHERE d.low_inference_notes IS NOT NULL AND LENGTH(d.low_inference_notes) > 0) as count,
    ROUND(COUNT(*) FILTER (WHERE d.low_inference_notes IS NOT NULL AND LENGTH(d.low_inference_notes) > 0) * 100.0 / COUNT(*), 1) as percentage
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id AND d.deleted_at IS NULL

UNION ALL

SELECT 
    'Has Framework ID' as metric,
    COUNT(*) FILTER (WHERE d.framework_id IS NOT NULL) as count,
    ROUND(COUNT(*) FILTER (WHERE d.framework_id IS NOT NULL) * 100.0 / COUNT(*), 1) as percentage
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id AND d.deleted_at IS NULL;

-- Potential duplicate evaluations
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.teacher_name,
    d.school_name,
    DATE(d.created_at) as evaluation_date,
    COUNT(*) as evaluation_count,
    ARRAY_AGG(d.id) as evaluation_ids,
    ARRAY_AGG(d.evaluator) as evaluators
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
GROUP BY d.teacher_name, d.school_name, DATE(d.created_at)
HAVING COUNT(*) > 1
ORDER BY evaluation_count DESC, d.teacher_name;

-- ============================================================================
-- PERFORMANCE AND INDEXING ANALYSIS
-- ============================================================================

-- Current index usage analysis
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'public' 
    AND tablename IN ('danielson', 'users', 'organizations')
ORDER BY tablename, indexname;

-- Table sizes and row counts
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    (SELECT COUNT(*) FROM public.danielson WHERE deleted_at IS NULL) as active_rows
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename = 'danielson'

UNION ALL

SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    (SELECT COUNT(*) FROM public.users WHERE deleted_at IS NULL) as active_rows
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename = 'users'

UNION ALL

SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    (SELECT COUNT(*) FROM public.organizations WHERE deleted_at IS NULL) as active_rows
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename = 'organizations';

-- ============================================================================
-- SAMPLE DATA EXPLORATION
-- ============================================================================

-- Sample evaluation JSON structures for analysis
WITH user_org AS (
    SELECT organization_id 
    FROM public.users 
    WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL
)
SELECT 
    d.id,
    d.teacher_name,
    d.school_name,
    d.framework_id,
    d.created_at,
    d.evaluation,
    d.ai_evaluation,
    SUBSTR(d.low_inference_notes, 1, 200) as notes_sample
FROM public.danielson d
CROSS JOIN user_org uo
WHERE d.organization_id = uo.organization_id
    AND d.deleted_at IS NULL
    AND d.evaluation IS NOT NULL
ORDER BY d.created_at DESC
LIMIT 5;

-- Organization structure and hierarchy
SELECT 
    o.id,
    o.name as organization_name,
    o.performance_level_config,
    COUNT(DISTINCT u.id) as total_users,
    COUNT(DISTINCT d.id) as total_evaluations,
    COUNT(DISTINCT d.school_name) as unique_schools,
    COUNT(DISTINCT d.teacher_name) as unique_teachers
FROM public.organizations o
LEFT JOIN public.users u ON o.id = u.organization_id AND u.deleted_at IS NULL
LEFT JOIN public.danielson d ON o.id = d.organization_id AND d.deleted_at IS NULL
WHERE o.deleted_at IS NULL
    AND o.id = (SELECT organization_id FROM public.users WHERE email = 'justin@swiftscore.org' AND deleted_at IS NULL)
GROUP BY o.id, o.name, o.performance_level_config;