#!/usr/bin/env python3
"""
Test script for database connection and basic functionality.

This script tests:
1. Database connection pooling
2. User permission loading
3. Basic evaluation queries
4. Caching functionality
5. Text chunking utilities
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import (
    get_database_pool, 
    close_database_pool,
    EvaluationQueries,
    get_evaluation_cache,
    DatabaseConnectionError
)
from utils import chunk_evaluation_notes, TextChunker


async def test_database_connection():
    """Test basic database connectivity."""
    print("Testing database connection...")
    
    try:
        pool = await get_database_pool()
        health_check = await pool.health_check()
        
        if health_check:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database health check failed")
            return False
            
    except DatabaseConnectionError as e:
        print(f"❌ Database connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def test_user_permissions(user_email: str = "justin@swiftscore.org"):
    """Test user permission loading."""
    print(f"Testing user permissions for {user_email}...")
    
    try:
        permissions = await EvaluationQueries.get_user_permissions(user_email)
        
        if permissions:
            print("✅ User permissions loaded successfully")
            print(f"   - User ID: {permissions.user_id}")
            print(f"   - Organization: {permissions.organization_id}")
            print(f"   - Role: {permissions.class_role or 'N/A'}")
            print(f"   - Schools: {len(permissions.school_ids)} accessible")
            print(f"   - Is Principal: {permissions.is_principal}")
            print(f"   - Is Superintendent: {permissions.is_superintendent}")
            return permissions
        else:
            print(f"❌ User {user_email} not found")
            return None
            
    except Exception as e:
        print(f"❌ Error loading user permissions: {e}")
        return None


async def test_evaluation_queries(permissions):
    """Test evaluation queries with permission filtering."""
    if not permissions:
        print("⏭️  Skipping evaluation queries - no permissions")
        return
        
    print("Testing evaluation queries...")
    
    try:
        # Test basic evaluation listing
        evaluations = await EvaluationQueries.get_evaluations_for_user(
            permissions, limit=5
        )
        
        print(f"✅ Found {len(evaluations)} evaluations")
        
        if evaluations:
            eval_sample = evaluations[0]
            print(f"   - Sample: {eval_sample.teacher_name} at {eval_sample.school_name}")
            print(f"   - Created: {eval_sample.created_at}")
            print(f"   - Notes length: {len(eval_sample.low_inference_notes or '')} chars")
            
            # Test single evaluation retrieval
            single_eval = await EvaluationQueries.get_evaluation_by_id(
                eval_sample.id, permissions
            )
            
            if single_eval:
                print("✅ Single evaluation retrieval works")
            else:
                print("❌ Single evaluation retrieval failed")
        
        # Test organizational queries
        teachers = await EvaluationQueries.get_teachers_for_user(permissions)
        schools = await EvaluationQueries.get_schools_for_user(permissions)
        org_tree = await EvaluationQueries.get_organization_tree(permissions)
        
        print(f"✅ Organizational queries successful:")
        print(f"   - Teachers: {len(teachers)}")
        print(f"   - Schools: {len(schools)}")
        print(f"   - Org tree has {org_tree.total_evaluations} total evaluations")
        
        return evaluations
        
    except Exception as e:
        print(f"❌ Error in evaluation queries: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_caching():
    """Test caching functionality."""
    print("Testing caching functionality...")
    
    try:
        cache = await get_evaluation_cache()
        
        # Test basic cache operations
        test_key = "test_eval_123"
        test_data = {"teacher": "John Doe", "score": 85}
        
        # Set and get
        await cache.set_evaluation(test_key, test_data)
        retrieved = await cache.get_evaluation(test_key)
        
        if retrieved == test_data:
            print("✅ Basic cache operations work")
        else:
            print("❌ Cache retrieval mismatch")
            
        # Test specialized caching
        await cache.set_teacher_summary("Jane Smith", "Test School", {"summary": "test"})
        teacher_summary = await cache.get_teacher_summary("Jane Smith", "Test School")
        
        if teacher_summary:
            print("✅ Teacher-level caching works")
        else:
            print("❌ Teacher-level caching failed")
            
        # Test cache stats
        stats = await cache.get_cache_stats()
        print(f"✅ Cache stats: {stats['total_entries']} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in caching tests: {e}")
        return False


def test_text_chunking():
    """Test text chunking utilities."""
    print("Testing text chunking...")
    
    try:
        # Test with sample evaluation notes
        sample_notes = """
        Domain IIa: Creating an Environment of Respect and Rapport
        
        Evidence: The teacher greeted each student by name as they entered the classroom. 
        Students were observed helping each other with materials and showing patience 
        when a classmate struggled with a concept. The teacher used encouraging language 
        throughout the lesson.
        
        Areas for Growth: Consider implementing more structured peer interaction 
        protocols to maximize learning opportunities.
        
        Domain IIb: Establishing a Culture for Learning
        
        Evidence: High expectations were evident through challenging questions and 
        rigorous academic tasks. Students demonstrated investment in their learning 
        through active participation and thoughtful responses.
        
        The classroom displayed student work prominently, and the teacher referenced 
        learning objectives multiple times during instruction.
        """
        
        # Test semantic chunking
        chunks = chunk_evaluation_notes(sample_notes, max_chunk_size=300)
        
        print(f"✅ Text chunking successful: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks):
            print(f"   - Chunk {i + 1}: {len(chunk.content)} chars, type: {chunk.chunk_type}")
            
        # Test parallel processing chunking
        from utils import chunk_for_parallel_processing
        parallel_chunks = chunk_for_parallel_processing(sample_notes, target_chunks=2)
        
        print(f"✅ Parallel chunking: {len(parallel_chunks)} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in text chunking: {e}")
        return False


async def run_all_tests():
    """Run all tests sequentially."""
    print("🚀 Starting database integration layer tests...\n")
    
    # Check environment
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable not set")
        print("   Please set DATABASE_URL to your Supabase connection string")
        return False
    
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Database connection
        if await test_database_connection():
            tests_passed += 1
        print()
        
        # Test 2: User permissions
        permissions = await test_user_permissions()
        if permissions:
            tests_passed += 1
        print()
        
        # Test 3: Evaluation queries  
        evaluations = await test_evaluation_queries(permissions)
        if evaluations is not None:
            tests_passed += 1
        print()
        
        # Test 4: Caching
        if await test_caching():
            tests_passed += 1
        print()
        
        # Test 5: Text chunking (doesn't require DB)
        if test_text_chunking():
            tests_passed += 1
        print()
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await close_database_pool()
    
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Database integration layer is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return False


if __name__ == "__main__":
    # Set up minimal environment if needed
    if not os.getenv('DATABASE_URL'):
        print("💡 Tip: Set DATABASE_URL environment variable for testing")
        print("   Example: DATABASE_URL=postgresql://user:pass@host:port/db")
        sys.exit(1)
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)