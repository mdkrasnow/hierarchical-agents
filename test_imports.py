"""Test imports to debug the package structure."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("Testing imports...")

try:
    print("Importing LLM utils...")
    from utils.llm import LLMClient, create_llm_client
    print("✓ LLM utils imported successfully")
    
    print("Importing template system...")
    from agents.templates import TemplateManager, get_template_manager, FileTemplateLoader
    print("✓ Template system imported successfully")
    
    print("Importing BaseAgent...")
    from agents.base import BaseAgent
    print("✓ BaseAgent imported successfully")
    
    print("Importing example agents...")
    from agents.examples import SimpleAnalyzerAgent
    print("✓ Example agents imported successfully")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()