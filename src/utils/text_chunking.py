"""
Text chunking utilities for large evaluation notes and content.

Provides intelligent text chunking for large low_inference_notes and other content
that needs to fit within LLM context limits while preserving semantic coherence.
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ChunkStrategy(Enum):
    """Different strategies for chunking text."""
    SENTENCE = "sentence"  # Split on sentence boundaries
    PARAGRAPH = "paragraph"  # Split on paragraph boundaries
    SEMANTIC = "semantic"  # Split based on content structure (for evaluations)
    FIXED_SIZE = "fixed_size"  # Fixed character/token chunks with overlap


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    token_count: Optional[int] = None
    chunk_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    max_chunk_size: int = 4000  # Maximum characters per chunk
    overlap_size: int = 200  # Overlap between chunks
    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    preserve_sentences: bool = True
    min_chunk_size: int = 100  # Minimum chunk size to avoid tiny fragments
    

class TextChunker:
    """
    Intelligent text chunking for evaluation content.
    
    Specialized for educational evaluation notes which often contain:
    - Domain/component headings
    - Observation notes
    - Feedback sections
    - Evidence descriptions
    """

    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize text chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Common patterns in evaluation notes
        self.domain_patterns = [
            r'(?i)domain\s+[ivx]+[a-z]?[.\-:]',  # Domain IIa, Domain IV-B, etc.
            r'(?i)component\s+[ivx]+[a-z]?[.\-:]',  # Component IIa
            r'(?i)rubric\s+[ivx]+[a-z]?[.\-:]',  # Rubric IIa
        ]
        
        self.section_patterns = [
            r'(?i)(evidence|observation|feedback|notes?|comments?)\s*:',
            r'(?i)(strengths?|areas?\s+for\s+growth|recommendations?)\s*:',
            r'(?i)(planning|instruction|classroom\s+environment|professional\s+responsibilities)\s*:',
        ]

    def chunk_text(self, text: str, source_type: str = "evaluation") -> List[TextChunk]:
        """
        Chunk text based on configuration and content type.
        
        Args:
            text: Text to chunk
            source_type: Type of source content ("evaluation", "notes", "general")
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean and normalize text
        text = self._preprocess_text(text)
        
        if len(text) <= self.config.max_chunk_size:
            return [TextChunk(
                content=text,
                chunk_index=0,
                start_position=0,
                end_position=len(text),
                chunk_type="complete",
                metadata={"source_type": source_type}
            )]
        
        # Choose chunking strategy based on content type and config
        if self.config.strategy == ChunkStrategy.SEMANTIC and source_type == "evaluation":
            return self._chunk_evaluation_semantic(text)
        elif self.config.strategy == ChunkStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text)
        elif self.config.strategy == ChunkStrategy.SENTENCE:
            return self._chunk_by_sentence(text)
        else:  # FIXED_SIZE
            return self._chunk_fixed_size(text)

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()

    def _chunk_evaluation_semantic(self, text: str) -> List[TextChunk]:
        """
        Chunk evaluation text based on semantic structure.
        
        Tries to preserve:
        - Domain/component sections together
        - Evidence with related observations
        - Complete feedback sections
        """
        chunks = []
        
        # First, try to identify domain/component boundaries
        domain_splits = self._find_domain_boundaries(text)
        
        if domain_splits:
            # Split by domains first, then chunk each domain if needed
            for i, (start, end, domain_name) in enumerate(domain_splits):
                domain_text = text[start:end]
                
                if len(domain_text) <= self.config.max_chunk_size:
                    # Domain fits in one chunk
                    chunks.append(TextChunk(
                        content=domain_text,
                        chunk_index=len(chunks),
                        start_position=start,
                        end_position=end,
                        chunk_type="domain",
                        metadata={"domain": domain_name}
                    ))
                else:
                    # Need to sub-chunk this domain
                    sub_chunks = self._chunk_by_sentence(domain_text)
                    for sub_chunk in sub_chunks:
                        sub_chunk.start_position += start
                        sub_chunk.end_position += start
                        sub_chunk.chunk_index = len(chunks)
                        sub_chunk.chunk_type = "domain_part"
                        sub_chunk.metadata = {"domain": domain_name}
                        chunks.append(sub_chunk)
        else:
            # No clear domain structure, fall back to paragraph/sentence chunking
            chunks = self._chunk_by_paragraph(text)
        
        return chunks

    def _find_domain_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find domain/component boundaries in evaluation text.
        
        Returns:
            List of (start_pos, end_pos, domain_name) tuples
        """
        boundaries = []
        
        # Look for domain patterns
        for pattern in self.domain_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append((match.start(), match.group()))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        if not boundaries:
            return []
        
        # Create segments between boundaries
        segments = []
        for i, (pos, name) in enumerate(boundaries):
            start_pos = pos
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            
            segments.append((start_pos, end_pos, name.strip()))
        
        return segments

    def _chunk_by_paragraph(self, text: str) -> List[TextChunk]:
        """Chunk text by paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, len(chunks), current_start, 
                        current_start + len(current_chunk), "paragraph"
                    ))
                    current_start = current_start + len(current_chunk) + 2  # +2 for \n\n
                
                # Start new chunk with current paragraph
                if len(paragraph) <= self.config.max_chunk_size:
                    current_chunk = paragraph
                else:
                    # Paragraph is too long, need to split further
                    para_chunks = self._chunk_by_sentence(paragraph)
                    for chunk in para_chunks:
                        chunk.start_position += current_start
                        chunk.end_position += current_start
                        chunk.chunk_index = len(chunks)
                        chunks.append(chunk)
                    current_chunk = ""
                    current_start += len(paragraph) + 2
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, len(chunks), current_start,
                current_start + len(current_chunk), "paragraph"
            ))
        
        return chunks

    def _chunk_by_sentence(self, text: str) -> List[TextChunk]:
        """Chunk text by sentence boundaries."""
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, len(chunks), current_start,
                        current_start + len(current_chunk), "sentence"
                    ))
                    current_start += len(current_chunk) + 1
                
                # Start new chunk
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, len(chunks), current_start,
                current_start + len(current_chunk), "sentence"
            ))
        
        return chunks

    def _chunk_fixed_size(self, text: str) -> List[TextChunk]:
        """Chunk text into fixed-size chunks with overlap."""
        chunks = []
        
        if not text:
            return chunks
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position
            end = min(start + self.config.max_chunk_size, len(text))
            
            # Try to break at word boundary if possible
            if end < len(text) and self.config.preserve_sentences:
                # Look for sentence boundary within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos
            
            chunk_content = text[start:end].strip()
            
            if chunk_content and len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_content, chunk_index, start, end, "fixed_size"
                ))
                chunk_index += 1
            
            # Calculate next start position with overlap
            if end >= len(text):
                break
                
            start = end - self.config.overlap_size
            if start < 0:
                start = 0
        
        return chunks

    def _create_chunk(self, content: str, index: int, start: int, end: int, chunk_type: str) -> TextChunk:
        """Create a TextChunk object with metadata."""
        return TextChunk(
            content=content,
            chunk_index=index,
            start_position=start,
            end_position=end,
            token_count=self._estimate_tokens(content),
            chunk_type=chunk_type,
            metadata={
                "length": len(content),
                "estimated_tokens": self._estimate_tokens(content)
            }
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count for text."""
        # Approximate: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def chunk_evaluation_notes(self, notes: str) -> List[TextChunk]:
        """Convenience method for chunking evaluation notes specifically."""
        return self.chunk_text(notes, "evaluation")

    def chunk_for_llm_context(self, text: str, max_tokens: int = 3000) -> List[TextChunk]:
        """
        Chunk text specifically for LLM context limits.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of chunks that fit within token limits
        """
        # Convert token limit to approximate character limit
        max_chars = max_tokens * 4  # Rough approximation
        
        config = ChunkingConfig(
            max_chunk_size=max_chars,
            overlap_size=min(200, max_chars // 10),
            strategy=ChunkStrategy.SEMANTIC,
            preserve_sentences=True
        )
        
        chunker = TextChunker(config)
        return chunker.chunk_text(text)


# Convenience functions
def chunk_evaluation_notes(notes: str, max_chunk_size: int = 4000) -> List[TextChunk]:
    """
    Quick function to chunk evaluation notes.
    
    Args:
        notes: Evaluation notes text
        max_chunk_size: Maximum size per chunk
        
    Returns:
        List of TextChunk objects
    """
    config = ChunkingConfig(
        max_chunk_size=max_chunk_size,
        strategy=ChunkStrategy.SEMANTIC
    )
    chunker = TextChunker(config)
    return chunker.chunk_evaluation_notes(notes)


def chunk_for_parallel_processing(text: str, target_chunks: int = 4) -> List[TextChunk]:
    """
    Chunk text for parallel processing by multiple agents.
    
    Args:
        text: Text to chunk
        target_chunks: Target number of chunks for parallel processing
        
    Returns:
        List of approximately equal-sized chunks
    """
    if not text:
        return []
    
    chunk_size = max(1000, len(text) // target_chunks)
    config = ChunkingConfig(
        max_chunk_size=chunk_size,
        overlap_size=chunk_size // 10,
        strategy=ChunkStrategy.SENTENCE,
        preserve_sentences=True
    )
    
    chunker = TextChunker(config)
    return chunker.chunk_text(text)