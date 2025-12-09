#!/usr/bin/env python3
"""
Question Dataset Explorer Script

This script provides a command-line interface for exploring and analyzing
the evaluation question dataset. It supports various filtering and display options.

Usage examples:
    python scripts/list_questions.py --all
    python scripts/list_questions.py --category hierarchical-agents
    python scripts/list_questions.py --difficulty advanced 
    python scripts/list_questions.py --tag architecture --format detailed
    python scripts/list_questions.py --sample 5
    python scripts/list_questions.py --stats
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoring import QuestionLoader, Question, QuestionDataset

def format_question_brief(question: Question) -> str:
    """Format a question for brief display."""
    return f"[{question.id}] ({question.category}/{question.difficulty}) {question.prompt[:100]}..."

def format_question_detailed(question: Question) -> str:
    """Format a question for detailed display."""
    lines = [
        f"ID: {question.id}",
        f"Category: {question.category}",
        f"Difficulty: {question.difficulty}",
        f"Tags: {', '.join(question.tags)}",
        f"Rubric Focus: {', '.join(question.rubric_focus)}",
        "",
        f"Prompt:",
        f"  {question.prompt}",
        "",
        f"Expected Key Points:",
    ]
    
    for point in question.expected_key_points:
        lines.append(f"  • {point}")
    
    lines.extend([
        "",
        f"Style Requirements:",
        f"  Format: {question.style_requirements.get('format', 'not specified')}",
        f"  Tone: {question.style_requirements.get('tone', 'not specified')}",
        f"  Length: {question.style_requirements.get('length', 'not specified')}",
        f"  Include Examples: {question.style_requirements.get('include_examples', False)}",
    ])
    
    if question.description:
        lines.extend(["", f"Description: {question.description}"])
    
    if question.notes:
        lines.extend(["", f"Notes: {question.notes}"])
    
    lines.append(f"\nCreated: {question.created_date}")
    
    return "\n".join(lines)

def format_question_json(question: Question) -> str:
    """Format a question as JSON."""
    return json.dumps(question.to_dict(), indent=2, ensure_ascii=False)

def print_questions(dataset: QuestionDataset, format_type: str = "brief"):
    """Print questions in the specified format."""
    if not dataset.questions:
        print("No questions found matching the criteria.")
        return
    
    print(f"\nFound {len(dataset.questions)} question(s):\n")
    
    for i, question in enumerate(dataset.questions):
        if format_type == "brief":
            print(f"{i+1:2}. {format_question_brief(question)}")
        elif format_type == "detailed":
            print(f"=== Question {i+1} ===")
            print(format_question_detailed(question))
            print()
        elif format_type == "json":
            print(format_question_json(question))
            if i < len(dataset.questions) - 1:  # Add separator between questions
                print()
    
    print()

def print_statistics(loader: QuestionLoader):
    """Print comprehensive dataset statistics."""
    stats = loader.get_dataset_statistics()
    
    print("=== Question Dataset Statistics ===\n")
    
    print(f"Total Questions: {stats['total_questions']}")
    print(f"Last Updated: {stats.get('last_updated', 'Unknown')}")
    print()
    
    # Categories
    print("Distribution by Category:")
    for category, count in sorted(stats['categories'].items()):
        percentage = (count / stats['total_questions']) * 100
        print(f"  {category:20} {count:3} ({percentage:5.1f}%)")
    print()
    
    # Difficulty levels
    print("Distribution by Difficulty:")
    for difficulty, count in sorted(stats['difficulties'].items()):
        percentage = (count / stats['total_questions']) * 100
        print(f"  {difficulty:20} {count:3} ({percentage:5.1f}%)")
    print()
    
    # Rubric focus
    if 'rubric_focus_distribution' in stats:
        print("Rubric Focus Distribution:")
        for focus, count in sorted(stats['rubric_focus_distribution'].items()):
            percentage = (count / stats['total_questions']) * 100
            print(f"  {focus:20} {count:3} ({percentage:5.1f}%)")
        print()
    
    # Most common tags
    print("Most Common Tags:")
    sorted_tags = sorted(stats['tags'].items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags[:15]:
        percentage = (count / stats['total_questions']) * 100
        print(f"  {tag:20} {count:3} ({percentage:5.1f}%)")
    
    # Tag co-occurrence
    if 'common_tag_pairs' in stats and stats['common_tag_pairs']:
        print("\nMost Common Tag Pairs:")
        for (tag1, tag2), count in stats['common_tag_pairs'][:10]:
            print(f"  {tag1} + {tag2:15} {count:3}")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Explore and analyze the evaluation question dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                              # List all questions (brief)
  %(prog)s --category hierarchical-agents    # Questions in specific category  
  %(prog)s --difficulty advanced             # Questions of specific difficulty
  %(prog)s --tag architecture                # Questions with specific tag
  %(prog)s --rubric-focus coverage           # Questions focusing on coverage
  %(prog)s --sample 5                        # Random sample of 5 questions
  %(prog)s --sample 3 --category sql         # 3 questions from SQL category
  %(prog)s --stats                           # Show dataset statistics
  %(prog)s --id hma-001 --format detailed    # Show specific question in detail
  %(prog)s --search "coordination" --format json  # Search and show as JSON
        """
    )
    
    # Selection filters
    selection = parser.add_argument_group('question selection')
    selection.add_argument('--all', action='store_true', 
                          help='Show all questions')
    selection.add_argument('--id', type=str,
                          help='Show specific question by ID')
    selection.add_argument('--category', type=str,
                          help='Filter by category (hierarchical-agents, supabase-sql, general-writing)')
    selection.add_argument('--difficulty', type=str, 
                          choices=['beginner', 'intermediate', 'advanced', 'expert'],
                          help='Filter by difficulty level')
    selection.add_argument('--tag', type=str,
                          help='Filter by tag')
    selection.add_argument('--rubric-focus', type=str,
                          choices=['coverage', 'detail_specificity', 'structure_coherence', 'style_tone', 'instruction_following'],
                          help='Filter by rubric focus dimension')
    selection.add_argument('--sample', type=int, metavar='N',
                          help='Random sample of N questions')
    selection.add_argument('--search', type=str,
                          help='Search questions by text in prompt, tags, or key points')
    
    # Display options
    display = parser.add_argument_group('display options')
    display.add_argument('--format', choices=['brief', 'detailed', 'json'], 
                        default='brief',
                        help='Output format (default: brief)')
    display.add_argument('--stats', action='store_true',
                        help='Show dataset statistics instead of questions')
    
    # Utility options  
    utility = parser.add_argument_group('utility options')
    utility.add_argument('--list-categories', action='store_true',
                        help='List all available categories')
    utility.add_argument('--list-tags', action='store_true',
                        help='List all available tags')
    utility.add_argument('--list-difficulties', action='store_true',
                        help='List all available difficulty levels')
    
    args = parser.parse_args()
    
    # Load the dataset
    try:
        loader = QuestionLoader()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Handle utility options
    if args.list_categories:
        print("Available categories:")
        for category in loader.get_available_categories():
            print(f"  {category}")
        return
    
    if args.list_tags:
        print("Available tags:")
        for tag in loader.get_available_tags():
            print(f"  {tag}")
        return
    
    if args.list_difficulties:
        print("Available difficulty levels:")
        for difficulty in loader.get_available_difficulties():
            print(f"  {difficulty}")
        return
    
    # Handle statistics
    if args.stats:
        print_statistics(loader)
        return
    
    # Handle question selection
    if args.id:
        question = loader.get_question_by_id(args.id)
        if question:
            dataset = QuestionDataset([question], loader.categories_config)
        else:
            print(f"Question with ID '{args.id}' not found.")
            return
    
    elif args.search:
        from scoring.loaders import search_questions
        dataset = search_questions(args.search)
    
    elif args.all:
        dataset = loader.get_all_questions()
    
    elif args.category:
        dataset = loader.filter_questions(category=args.category)
    
    elif args.difficulty:
        dataset = loader.filter_questions(difficulty=args.difficulty)
    
    elif args.tag:
        dataset = loader.filter_questions(tags=args.tag)
    
    elif args.rubric_focus:
        dataset = loader.filter_questions(rubric_focus=args.rubric_focus)
    
    elif args.sample:
        filters = {}
        if args.category:
            filters['category'] = args.category
        if args.difficulty:
            filters['difficulty'] = args.difficulty
        if args.tag:
            filters['tags'] = args.tag
        if args.rubric_focus:
            filters['rubric_focus'] = args.rubric_focus
            
        dataset = loader.dataset.sample_questions(args.sample, **filters)
    
    else:
        # No specific selection, show brief help
        print("No selection criteria specified. Use --help for options.")
        print("\nQuick examples:")
        print("  --all              # Show all questions")
        print("  --stats            # Show statistics") 
        print("  --sample 5         # Show 5 random questions")
        print("  --category sql     # Show SQL questions")
        return
    
    # Display the selected questions
    print_questions(dataset, args.format)

if __name__ == "__main__":
    main()