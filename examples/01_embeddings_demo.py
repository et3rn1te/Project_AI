"""
Embeddings Demo - Understanding How Text Becomes Numbers

This script demonstrates:
1. How text is converted into embeddings (vectors)
2. How similar sentences have similar embeddings
3. How to measure similarity between embeddings

Run this to see the "magic" behind RAG!
"""

from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate how similar two vectors are (0 to 1)
    1.0 = identical, 0.0 = completely different
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def main():
    print("=" * 60)
    print("🧠 EMBEDDINGS DEMO - How Text Becomes Searchable")
    print("=" * 60)
    print()
    
    # Load the embedding model (same one we'll use in RAG)
    print("📥 Loading embedding model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!\n")
    
    # Example sentences about AI/ML
    sentences = [
        "Neural networks are inspired by the human brain",
        "Deep learning uses artificial neural networks",
        "What should I eat for dinner tonight?",
        "Machine learning is a subset of artificial intelligence",
        "The weather is nice today"
    ]
    
    print("📝 Example Sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"   {i}. {sentence}")
    print()
    
    # Generate embeddings
    print("🔄 Converting sentences to embeddings (vectors)...")
    embeddings = model.encode(sentences)
    print(f"✅ Created {len(embeddings)} embeddings")
    print(f"   Each embedding has {len(embeddings[0])} dimensions\n")
    
    # Show what an embedding looks like
    print("🔍 What does an embedding look like?")
    print(f"   First sentence: '{sentences[0]}'")
    print(f"   Embedding (first 10 numbers): {embeddings[0][:10]}")
    print(f"   ... and {len(embeddings[0]) - 10} more numbers!\n")
    
    # Calculate similarities
    print("=" * 60)
    print("📊 SIMILARITY ANALYSIS")
    print("=" * 60)
    print()
    
    # Compare sentence 1 with all others
    reference_sentence = sentences[0]
    reference_embedding = embeddings[0]
    
    print(f"🎯 Comparing everything to: '{reference_sentence}'\n")
    
    similarities = []
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        similarity = cosine_similarity(reference_embedding, embedding)
        similarities.append((sentence, similarity))
        
        # Visual similarity bar
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"   {i+1}. [{bar}] {similarity:.3f}")
        print(f"      '{sentence}'")
        print()
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("=" * 60)
    print("🏆 RANKED BY SIMILARITY")
    print("=" * 60)
    print()
    
    for i, (sentence, similarity) in enumerate(similarities, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {i}. {similarity:.3f} - '{sentence}'")
    
    print()
    print("=" * 60)
    print("💡 KEY INSIGHTS")
    print("=" * 60)
    print()
    print("✅ Sentences about AI/ML have HIGH similarity (0.7-1.0)")
    print("✅ Unrelated sentences have LOW similarity (0.0-0.3)")
    print("✅ This is how RAG finds relevant documents!")
    print()
    print("When you ask a question, RAG:")
    print("  1. Converts your question to an embedding")
    print("  2. Finds documents with similar embeddings")
    print("  3. Uses those documents to answer your question")
    print()

if __name__ == "__main__":
    main()
