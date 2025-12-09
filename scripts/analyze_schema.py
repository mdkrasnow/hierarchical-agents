#!/usr/bin/env python3
"""
Database schema analysis script for Supabase PostgreSQL.

This script analyzes the public.danielson evaluation JSON structure, 
examines ai_evaluation JSONB, inspects low_inference_notes patterns,
and explores the organizational hierarchy for a specific user.

Focus areas:
- Danielson evaluation JSON structure analysis
- AI evaluation JSONB field inspection  
- Low inference notes pattern analysis
- Organizations/schools/teachers relationship mapping
- Data quality assessment and indexing opportunities
- Evaluation density patterns over time
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import (
    get_database_pool, 
    close_database_pool,
    EvaluationQueries,
    DatabaseConnectionError
)


class SchemaAnalyzer:
    """Database schema analysis for hierarchical agents system."""
    
    def __init__(self):
        self.pool = None
        self.analysis_results = {}
        
    async def initialize(self):
        """Initialize database connection."""
        self.pool = await get_database_pool()
        
    async def analyze_table_structure(self):
        """Analyze the basic structure of key tables."""
        print("📊 Analyzing table structures...")
        
        # Get table information for key tables
        tables_query = """
        SELECT 
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
            AND table_name IN ('danielson', 'users', 'organizations', 'roles')
        ORDER BY table_name, ordinal_position
        """
        
        results = await self.pool.execute_query(tables_query)
        
        # Group by table
        tables = defaultdict(list)
        for row in results:
            tables[row['table_name']].append({
                'column': row['column_name'],
                'type': row['data_type'],
                'nullable': row['is_nullable'] == 'YES',
                'default': row['column_default']
            })
        
        self.analysis_results['table_structures'] = dict(tables)
        return tables
    
    async def analyze_danielson_evaluations(self, limit: int = 1000):
        """Analyze Danielson evaluation JSON structures in detail."""
        print(f"🔍 Analyzing {limit} Danielson evaluations...")
        
        # Get sample of evaluations with their JSON structures
        query = """
        SELECT 
            id,
            teacher_name,
            school_name,
            organization_id,
            evaluation,
            ai_evaluation,
            low_inference_notes,
            framework_id,
            is_informal,
            is_archived,
            created_at,
            LENGTH(evaluation::text) as eval_json_size,
            LENGTH(ai_evaluation::text) as ai_eval_json_size,
            LENGTH(low_inference_notes) as notes_length
        FROM public.danielson 
        WHERE deleted_at IS NULL 
            AND evaluation IS NOT NULL
        ORDER BY created_at DESC 
        LIMIT $1
        """
        
        results = await self.pool.execute_query(query, limit)
        
        # Analysis containers
        eval_structures = defaultdict(int)
        ai_eval_structures = defaultdict(int)
        domain_patterns = Counter()
        framework_usage = Counter()
        size_stats = {
            'eval_json_sizes': [],
            'ai_eval_sizes': [], 
            'notes_lengths': []
        }
        
        missing_data_counts = {
            'missing_evaluation': 0,
            'missing_ai_evaluation': 0,
            'missing_notes': 0,
            'missing_framework': 0
        }
        
        # Analyze each evaluation
        for row in results:
            # Track framework usage
            if row['framework_id']:
                framework_usage[row['framework_id']] += 1
            else:
                missing_data_counts['missing_framework'] += 1
                
            # Track data completeness
            if not row['evaluation']:
                missing_data_counts['missing_evaluation'] += 1
            if not row['ai_evaluation']:
                missing_data_counts['missing_ai_evaluation'] += 1
            if not row['low_inference_notes']:
                missing_data_counts['missing_notes'] += 1
                
            # Size tracking
            if row['eval_json_size']:
                size_stats['eval_json_sizes'].append(row['eval_json_size'])
            if row['ai_eval_json_size']:
                size_stats['ai_eval_sizes'].append(row['ai_eval_json_size'])
            if row['notes_length']:
                size_stats['notes_lengths'].append(row['notes_length'])
            
            # Analyze evaluation JSON structure
            if row['evaluation']:
                try:
                    eval_json = row['evaluation'] if isinstance(row['evaluation'], dict) else json.loads(row['evaluation'])
                    
                    # Track top-level structure
                    structure_key = tuple(sorted(eval_json.keys()))
                    eval_structures[structure_key] += 1
                    
                    # Look for domain patterns
                    for key in eval_json.keys():
                        if re.match(r'^[IVX]+[A-Z]?-[A-Z]$', key):  # Roman numeral domains like "I-A", "II-B"
                            domain_patterns[key] += 1
                        elif 'domain' in key.lower():
                            domain_patterns[key] += 1
                            
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️  JSON parsing error for evaluation {row['id']}: {e}")
                    
            # Analyze AI evaluation structure
            if row['ai_evaluation']:
                try:
                    ai_eval_json = row['ai_evaluation'] if isinstance(row['ai_evaluation'], dict) else json.loads(row['ai_evaluation'])
                    structure_key = tuple(sorted(ai_eval_json.keys()))
                    ai_eval_structures[structure_key] += 1
                    
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️  AI evaluation JSON parsing error for {row['id']}: {e}")
        
        # Calculate statistics
        def calc_stats(values):
            if not values:
                return {}
            values.sort()
            n = len(values)
            return {
                'count': n,
                'min': min(values),
                'max': max(values), 
                'avg': sum(values) / n,
                'median': values[n//2],
                'p95': values[int(n * 0.95)]
            }
        
        size_statistics = {
            'evaluation_json': calc_stats(size_stats['eval_json_sizes']),
            'ai_evaluation_json': calc_stats(size_stats['ai_eval_sizes']),
            'notes_text': calc_stats(size_stats['notes_lengths'])
        }
        
        self.analysis_results['danielson_analysis'] = {
            'total_analyzed': len(results),
            'evaluation_structures': dict(eval_structures),
            'ai_evaluation_structures': dict(ai_eval_structures),
            'domain_patterns': dict(domain_patterns),
            'framework_usage': dict(framework_usage),
            'missing_data': missing_data_counts,
            'size_statistics': size_statistics
        }
        
        return self.analysis_results['danielson_analysis']
    
    async def analyze_user_access_patterns(self, target_email: str = "justin@swiftscore.org"):
        """Analyze organizational structure and access patterns for a specific user."""
        print(f"👤 Analyzing access patterns for {target_email}...")
        
        # Get user permissions
        permissions = await EvaluationQueries.get_user_permissions(target_email)
        if not permissions:
            print(f"❌ User {target_email} not found")
            return None
            
        # Get organizational tree
        org_tree = await EvaluationQueries.get_organization_tree(permissions)
        
        # Get detailed breakdowns
        teachers = await EvaluationQueries.get_teachers_for_user(permissions)
        schools = await EvaluationQueries.get_schools_for_user(permissions)
        
        # Analyze evaluation distribution
        evaluations = await EvaluationQueries.get_evaluations_for_user(
            permissions, limit=500
        )
        
        # Time-based analysis
        eval_by_month = defaultdict(int)
        eval_by_school = defaultdict(int)
        eval_by_teacher = defaultdict(int)
        
        for eval_record in evaluations:
            month_key = eval_record.created_at.strftime('%Y-%m')
            eval_by_month[month_key] += 1
            eval_by_school[eval_record.school_name] += 1
            eval_by_teacher[eval_record.teacher_name] += 1
            
        self.analysis_results['user_access_analysis'] = {
            'target_user': target_email,
            'permissions': {
                'organization_id': permissions.organization_id,
                'role': permissions.class_role,
                'school_access': len(permissions.school_ids),
                'is_principal': permissions.is_principal,
                'is_superintendent': permissions.is_superintendent,
                'is_evaluator': permissions.is_evaluator
            },
            'organizational_scope': {
                'organization_name': org_tree.organization_name,
                'accessible_schools': len(org_tree.accessible_schools),
                'accessible_teachers': len(org_tree.accessible_teachers),
                'total_evaluations': org_tree.total_evaluations
            },
            'distribution_patterns': {
                'evaluations_by_month': dict(eval_by_month),
                'evaluations_by_school': dict(eval_by_school),
                'evaluations_by_teacher': dict(eval_by_teacher),
                'teacher_summary': len(teachers),
                'school_summary': len(schools)
            }
        }
        
        return self.analysis_results['user_access_analysis']
    
    async def analyze_low_inference_notes(self, limit: int = 500):
        """Analyze patterns in low_inference_notes text fields."""
        print(f"📝 Analyzing low inference notes patterns ({limit} samples)...")
        
        query = """
        SELECT 
            id,
            teacher_name,
            school_name,
            low_inference_notes,
            LENGTH(low_inference_notes) as notes_length,
            created_at
        FROM public.danielson 
        WHERE deleted_at IS NULL 
            AND low_inference_notes IS NOT NULL 
            AND LENGTH(low_inference_notes) > 0
        ORDER BY created_at DESC 
        LIMIT $1
        """
        
        results = await self.pool.execute_query(query, limit)
        
        # Analysis patterns
        length_distribution = []
        word_counts = []
        common_phrases = Counter()
        domain_mentions = Counter()
        quality_indicators = {
            'has_evidence_keyword': 0,
            'has_domain_reference': 0,
            'has_specific_examples': 0,
            'has_recommendations': 0
        }
        
        # Domain patterns to look for
        domain_pattern = re.compile(r'\b[IVX]+[A-Z]?[-:]?[A-Z]?\b')
        evidence_keywords = ['evidence', 'observed', 'demonstrated', 'exhibited', 'showed']
        recommendation_keywords = ['recommend', 'suggest', 'consider', 'growth', 'improve']
        
        for row in results:
            notes = row['low_inference_notes']
            if not notes:
                continue
                
            length_distribution.append(row['notes_length'])
            word_count = len(notes.split())
            word_counts.append(word_count)
            
            # Convert to lowercase for analysis
            notes_lower = notes.lower()
            
            # Check for quality indicators
            if any(keyword in notes_lower for keyword in evidence_keywords):
                quality_indicators['has_evidence_keyword'] += 1
                
            if any(keyword in notes_lower for keyword in recommendation_keywords):
                quality_indicators['has_recommendations'] += 1
                
            # Look for domain references
            domain_matches = domain_pattern.findall(notes)
            if domain_matches:
                quality_indicators['has_domain_reference'] += 1
                for domain in domain_matches:
                    domain_mentions[domain] += 1
                    
            # Look for specific examples (sentences with specific teacher/student names or concrete actions)
            if 'teacher' in notes_lower and ('student' in notes_lower or 'classroom' in notes_lower):
                quality_indicators['has_specific_examples'] += 1
                
            # Extract common phrases (3-grams)
            words = notes_lower.split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if len(phrase) > 15:  # Ignore very short phrases
                    common_phrases[phrase] += 1
        
        # Calculate statistics
        def calc_stats(values):
            if not values:
                return {}
            values.sort()
            n = len(values)
            return {
                'count': n,
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / n,
                'median': values[n//2],
                'p95': values[int(n * 0.95)] if n > 20 else values[-1]
            }
            
        self.analysis_results['notes_analysis'] = {
            'total_analyzed': len(results),
            'length_statistics': calc_stats(length_distribution),
            'word_count_statistics': calc_stats(word_counts),
            'quality_indicators': quality_indicators,
            'domain_mentions': dict(domain_mentions.most_common(20)),
            'common_phrases': dict(common_phrases.most_common(15)),
            'quality_percentages': {
                key: round((value / len(results)) * 100, 2) 
                for key, value in quality_indicators.items()
            }
        }
        
        return self.analysis_results['notes_analysis']
    
    async def analyze_indexing_opportunities(self):
        """Analyze database for indexing opportunities and performance."""
        print("⚡ Analyzing indexing opportunities...")
        
        # Check current indexes
        indexes_query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE schemaname = 'public' 
            AND tablename IN ('danielson', 'users', 'organizations')
        ORDER BY tablename, indexname
        """
        
        current_indexes = await self.pool.execute_query(indexes_query)
        
        # Analyze query patterns that could benefit from indexing
        # Common filter patterns based on the queries we use
        query_patterns = {
            'danielson_by_org_and_date': {
                'columns': ['organization_id', 'created_at'],
                'rationale': 'Most queries filter by organization and sort by date'
            },
            'danielson_by_org_and_school': {
                'columns': ['organization_id', 'school_name'],
                'rationale': 'Principal-level access filtering'
            },
            'danielson_by_teacher': {
                'columns': ['organization_id', 'teacher_name', 'created_at'],
                'rationale': 'Teacher-specific evaluation lookups'
            },
            'danielson_created_by': {
                'columns': ['created_by', 'organization_id'],
                'rationale': 'Evaluator access filtering'
            },
            'users_by_email': {
                'columns': ['email', 'deleted_at'],
                'rationale': 'User authentication and lookup'
            },
            'jsonb_evaluation_domains': {
                'columns': ['(evaluation)', 'organization_id'],
                'rationale': 'JSON queries on evaluation structure',
                'type': 'GIN'
            }
        }
        
        # Check table sizes for index impact assessment
        size_query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public' 
            AND tablename IN ('danielson', 'users', 'organizations')
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        
        table_sizes = await self.pool.execute_query(size_query)
        
        self.analysis_results['indexing_analysis'] = {
            'current_indexes': [dict(row) for row in current_indexes],
            'recommended_indexes': query_patterns,
            'table_sizes': [dict(row) for row in table_sizes]
        }
        
        return self.analysis_results['indexing_analysis']
    
    async def generate_data_quality_report(self):
        """Generate comprehensive data quality assessment."""
        print("📋 Generating data quality report...")
        
        # Check for common data quality issues
        quality_checks = {}
        
        # 1. Null/missing data analysis
        null_analysis_query = """
        SELECT 
            'danielson' as table_name,
            COUNT(*) as total_rows,
            COUNT(*) - COUNT(teacher_name) as missing_teacher_name,
            COUNT(*) - COUNT(school_name) as missing_school_name,
            COUNT(*) - COUNT(evaluation) as missing_evaluation,
            COUNT(*) - COUNT(framework_id) as missing_framework_id,
            COUNT(*) - COUNT(low_inference_notes) as missing_notes,
            COUNT(*) - COUNT(ai_evaluation) as missing_ai_evaluation
        FROM public.danielson 
        WHERE deleted_at IS NULL
        """
        
        null_results = await self.pool.execute_query_one(null_analysis_query)
        quality_checks['missing_data'] = dict(null_results)
        
        # 2. Duplicate detection
        duplicate_query = """
        SELECT 
            teacher_name,
            school_name,
            DATE(created_at) as eval_date,
            COUNT(*) as duplicate_count
        FROM public.danielson 
        WHERE deleted_at IS NULL
        GROUP BY teacher_name, school_name, DATE(created_at)
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        LIMIT 20
        """
        
        duplicates = await self.pool.execute_query(duplicate_query)
        quality_checks['potential_duplicates'] = [dict(row) for row in duplicates]
        
        # 3. Data consistency checks
        consistency_query = """
        SELECT 
            organization_id,
            COUNT(DISTINCT school_name) as unique_schools,
            COUNT(DISTINCT teacher_name) as unique_teachers,
            COUNT(*) as total_evaluations
        FROM public.danielson 
        WHERE deleted_at IS NULL
        GROUP BY organization_id
        ORDER BY total_evaluations DESC
        """
        
        consistency = await self.pool.execute_query(consistency_query)
        quality_checks['organization_consistency'] = [dict(row) for row in consistency]
        
        # 4. Temporal distribution analysis
        temporal_query = """
        SELECT 
            DATE_TRUNC('month', created_at) as month,
            COUNT(*) as evaluations_count,
            COUNT(DISTINCT teacher_name) as unique_teachers,
            COUNT(DISTINCT school_name) as unique_schools
        FROM public.danielson 
        WHERE deleted_at IS NULL 
            AND created_at >= CURRENT_DATE - INTERVAL '24 months'
        GROUP BY DATE_TRUNC('month', created_at)
        ORDER BY month DESC
        """
        
        temporal = await self.pool.execute_query(temporal_query)
        quality_checks['temporal_distribution'] = [dict(row) for row in temporal]
        
        self.analysis_results['data_quality'] = quality_checks
        return quality_checks
    
    async def run_full_analysis(self, target_user: str = "justin@swiftscore.org"):
        """Run comprehensive database analysis."""
        print("🚀 Starting comprehensive database schema analysis...\n")
        
        try:
            await self.initialize()
            
            # Run all analysis components
            await self.analyze_table_structure()
            await self.analyze_danielson_evaluations(limit=1000)
            await self.analyze_user_access_patterns(target_user)
            await self.analyze_low_inference_notes(limit=500)
            await self.analyze_indexing_opportunities()
            await self.generate_data_quality_report()
            
            # Add metadata
            self.analysis_results['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'target_user': target_user,
                'database_connection': 'successful'
            }
            
            print("\n✅ Analysis complete!")
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            await close_database_pool()


def print_summary(results: Dict[str, Any]):
    """Print a summary of analysis results."""
    if not results:
        return
        
    print("\n" + "="*60)
    print("📊 DATABASE SCHEMA ANALYSIS SUMMARY")
    print("="*60)
    
    # Table structures
    if 'table_structures' in results:
        print(f"\n📋 Table Structures:")
        for table, columns in results['table_structures'].items():
            print(f"   {table}: {len(columns)} columns")
            
    # Danielson analysis
    if 'danielson_analysis' in results:
        da = results['danielson_analysis']
        print(f"\n🎯 Danielson Evaluations Analysis:")
        print(f"   Total analyzed: {da['total_analyzed']}")
        print(f"   Unique evaluation structures: {len(da['evaluation_structures'])}")
        print(f"   Domain patterns found: {len(da['domain_patterns'])}")
        print(f"   Framework usage: {len(da['framework_usage'])} frameworks")
        
        if 'missing_data' in da:
            md = da['missing_data']
            print(f"   Missing data:")
            for key, count in md.items():
                if count > 0:
                    print(f"     - {key}: {count}")
                    
    # User access
    if 'user_access_analysis' in results:
        ua = results['user_access_analysis']
        print(f"\n👤 User Access Analysis ({ua['target_user']}):")
        print(f"   Organization: {ua['organizational_scope']['organization_name']}")
        print(f"   Schools accessible: {ua['organizational_scope']['accessible_schools']}")
        print(f"   Teachers accessible: {ua['organizational_scope']['accessible_teachers']}")
        print(f"   Total evaluations: {ua['organizational_scope']['total_evaluations']}")
        
    # Notes analysis
    if 'notes_analysis' in results:
        na = results['notes_analysis']
        print(f"\n📝 Notes Analysis:")
        print(f"   Notes analyzed: {na['total_analyzed']}")
        if 'quality_percentages' in na:
            for indicator, percentage in na['quality_percentages'].items():
                print(f"   {indicator}: {percentage}%")
                
    # Data quality
    if 'data_quality' in results:
        dq = results['data_quality']
        if 'missing_data' in dq:
            total_rows = dq['missing_data']['total_rows']
            print(f"\n🔍 Data Quality (from {total_rows} evaluations):")
            for field, missing_count in dq['missing_data'].items():
                if 'missing_' in field and missing_count > 0:
                    percentage = round((missing_count / total_rows) * 100, 1)
                    print(f"   {field}: {missing_count} ({percentage}%)")


async def main():
    """Main analysis entry point."""
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable not set")
        print("   Please set DATABASE_URL to your Supabase connection string")
        return False
        
    target_user = sys.argv[1] if len(sys.argv) > 1 else "justin@swiftscore.org"
    
    analyzer = SchemaAnalyzer()
    results = await analyzer.run_full_analysis(target_user)
    
    if results:
        print_summary(results)
        
        # Save detailed results to JSON file
        output_file = Path(__file__).parent / "schema_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}")
        return True
    else:
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)