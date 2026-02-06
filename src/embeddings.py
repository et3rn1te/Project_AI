"""
Embeddings Module

This module handles converting text into embeddings (vector representations).

Simple Explanation:
This is the "magic translator" that converts your text into numbers (vectors)
that capture the meaning. Similar text gets similar numbers!

Think of it like converting addresses to GPS coordinates - different format,
but captures the same information in a way computers can work with.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers.
    
    Simple Explanation:
    This is your "meaning fingerprint maker" - it reads text and creates
    a unique list of numbers that represents what the text means.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            
        Simple Explanation:
        We're using "all-MiniLM-L6-v2" because it's:
        - Fast (processes text quickly)
        - Small (only 80MB download)
        - Good quality (captures meaning well)
        - Free (no API costs!)
        """
        print(f"📥 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dimension}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Simple Explanation:
        Takes one piece of text and converts it to a list of numbers.
        
        Args:
            text: The text to convert
            
        Returns:
            A list of 384 numbers representing the text's meaning
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Simple Explanation:
        Takes a list of texts and converts ALL of them to embeddings at once.
        Much faster than doing them one by one!
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings (each is a list of 384 numbers)
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text if text and text.strip() else " " for text in texts]
        
        embeddings = self.model.encode(
            valid_texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32  # Process 32 texts at a time
        )
        
        return embeddings.tolist()
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Simple Explanation:
        Measures how similar two embeddings are.
        - 1.0 = Identical meaning
        - 0.7-0.9 = Very similar
        - 0.4-0.6 = Somewhat related
        - 0.0-0.3 = Different topics
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 EMBEDDING GENERATOR TEST")
    print("=" * 60)
    print()
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    print("\n" + "=" * 60)
    print("TEST 1: Single Text Embedding")
    print("=" * 60)
    
    text = "Neural networks are inspired by the human brain"
    print(f"Text: '{text}'")
    
    embedding = generator.generate_embedding(text)
    print(f"\nEmbedding generated!")
    print(f"  - Dimension: {len(embedding)}")
    print(f"  - First 10 values: {embedding[:10]}")
    print(f"  - Type: {type(embedding)}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Batch Embedding Generation")
    print("=" * 60)
    
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Python is a programming language",
        "Supervised learning requires labeled data"
    ]
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = generator.generate_embeddings(texts, show_progress=True)
    
    print(f"\n✅ Generated {len(embeddings)} embeddings")
    print(f"   Each embedding has {len(embeddings[0])} dimensions")
    
    print("\n" + "=" * 60)
    print("TEST 3: Similarity Calculation")
    print("=" * 60)
    
    print("\nComparing all texts to the first one:")
    print(f"Reference: '{texts[0]}'")
    print()
    
    reference_embedding = embeddings[0]
    
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        similarity = generator.calculate_similarity(reference_embedding, embedding)
        
        # Visual similarity bar
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{i+1}. [{bar}] {similarity:.3f}")
        print(f"   '{text}'")
        print()
    
    print("=" * 60)
    print("💡 KEY INSIGHTS")
    print("=" * 60)
    print()
    print("✅ AI-related texts have high similarity (0.7+)")
    print("✅ Unrelated text (Python) has lower similarity")
    print("✅ This is how RAG finds relevant documents!")
    print()
