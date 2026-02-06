"""
Text Splitter Module

This module splits large documents into smaller chunks for better RAG performance.

Simple Explanation:
Imagine you have a huge textbook. Instead of trying to search the entire book at once,
we break it into chapters or sections. This makes it easier to find exactly what you need!

Why chunking?
- Embeddings work better on focused content
- LLMs have context limits
- Smaller chunks = more precise retrieval
"""

from typing import List
from src.document_loader import Document


class TextSplitter:
    """
    Splits documents into smaller, overlapping chunks.
    
    Simple Explanation:
    Like cutting a long rope into smaller pieces, but making sure each piece
    overlaps a bit with the next one so we don't lose context at the boundaries!
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Simple Explanation:
        - chunk_size: How long each piece should be (like 1 paragraph)
        - chunk_overlap: How much to repeat from the previous piece
                        (so we don't lose context at the edges)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Simple Explanation:
        Takes a long piece of text and breaks it into smaller pieces,
        making sure each piece overlaps with the next one a bit.
        
        Example:
            Text: "ABCDEFGHIJ" (chunk_size=4, overlap=2)
            Chunks: ["ABCD", "CDEF", "EFGH", "GHIJ"]
                     ^^^^    ^^^^    ^^^^    ^^^^
                       ^^      ^^      ^^  (overlaps)
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # Get the chunk
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position (accounting for overlap)
            start += self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loop if we're at the end
            if start >= text_length:
                break
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks, preserving metadata.
        
        Simple Explanation:
        Takes your documents (like book pages) and breaks each one into
        smaller pieces, but remembers where each piece came from!
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of new Document objects (chunks) with updated metadata
        """
        chunked_documents = []
        
        for doc in documents:
            # Split the text content
            text_chunks = self.split_text(doc.content)
            
            # Create new Document objects for each chunk
            for i, chunk_text in enumerate(text_chunks):
                # Copy original metadata and add chunk info
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(text_chunks)
                
                chunk_doc = Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                
                chunked_documents.append(chunk_doc)
        
        return chunked_documents


class SmartTextSplitter(TextSplitter):
    """
    A smarter text splitter that tries to split at natural boundaries.
    
    Simple Explanation:
    Instead of cutting text in the middle of a sentence (like cutting a word in half),
    this tries to split at paragraph breaks or sentence endings.
    Much better for reading and understanding!
    """
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text at natural boundaries (paragraphs, sentences).
        
        Simple Explanation:
        Tries to split at:
        1. Paragraph breaks (double newlines)
        2. Sentence endings (periods, question marks)
        3. If needed, falls back to character count
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # First, try splitting by paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap from previous
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + paragraph
                else:
                    # Paragraph itself is too long, split by sentences
                    sentences = self._split_by_sentences(paragraph)
                    current_chunk = self._merge_sentences(sentences)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple version)."""
        # Simple sentence splitting (can be improved with NLP libraries)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_sentences(self, sentences: List[str]) -> str:
        """Merge sentences up to chunk_size."""
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    # Add overlap
                    overlap = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                    current = overlap + " " + sentence
                else:
                    # Single sentence is too long, just add it
                    chunks.append(sentence)
                    current = ""
            else:
                current += " " + sentence if current else sentence
        
        if current:
            chunks.append(current.strip())
        
        return "\n".join(chunks)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("✂️  TEXT SPLITTER TEST")
    print("=" * 60)
    print()
    
    # Test with sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    
    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. The key idea is that machines can learn from data, identify patterns, and make decisions with minimal human intervention.
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks can learn complex patterns in large amounts of data. Deep learning has been particularly successful in areas such as computer vision, natural language processing, and speech recognition.
    """
    
    # Test basic splitter
    print("1. Basic Text Splitter (chunk_size=200, overlap=50)")
    print("-" * 60)
    
    basic_splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    basic_chunks = basic_splitter.split_text(sample_text)
    
    print(f"Created {len(basic_chunks)} chunks:\n")
    for i, chunk in enumerate(basic_chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"{chunk[:100]}...")
        print()
    
    # Test smart splitter
    print("\n" + "=" * 60)
    print("2. Smart Text Splitter (chunk_size=300, overlap=50)")
    print("-" * 60)
    
    smart_splitter = SmartTextSplitter(chunk_size=300, chunk_overlap=50)
    smart_chunks = smart_splitter.split_text(sample_text)
    
    print(f"Created {len(smart_chunks)} chunks:\n")
    for i, chunk in enumerate(smart_chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"{chunk[:150]}...")
        print()
    
    print("=" * 60)
    print("💡 Notice how Smart Splitter tries to keep paragraphs together!")
    print("=" * 60)
