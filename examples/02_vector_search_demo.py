"""
Vector Database Demo - Understanding ChromaDB

This script demonstrates:
1. How to store documents in ChromaDB
2. How semantic search works (meaning-based, not keyword-based)
3. How to retrieve relevant documents

This is the "smart librarian" in action!
"""

import chromadb
from chromadb.utils import embedding_functions

def main():
    print("=" * 60)
    print("📚 VECTOR DATABASE DEMO - The Smart Librarian")
    print("=" * 60)
    print()
    
    # Create a temporary in-memory database
    print("🔧 Creating ChromaDB instance...")
    client = chromadb.Client()
    
    # Set up embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create a collection (like a table in a database)
    print("📁 Creating collection 'ai_learning'...")
    collection = client.create_collection(
        name="ai_learning",
        embedding_function=embedding_function
    )
    print("✅ Collection created!\n")
    
    # Sample AI learning documents
    documents = [
        "Neural networks consist of layers of interconnected nodes that process information.",
        "Supervised learning requires labeled training data to learn patterns.",
        "Unsupervised learning finds patterns in data without labels.",
        "Deep learning uses multiple layers to learn hierarchical representations.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "The best pizza toppings are pepperoni and mushrooms.",
        "Python is a popular programming language for machine learning.",
        "Gradient descent is an optimization algorithm used in training models."
    ]
    
    print("📝 Adding 10 AI learning documents to the database...")
    
    # Add documents to the collection
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"topic": "AI" if i < 9 else "food"} for i in range(len(documents))]
    )
    
    print(f"✅ Added {len(documents)} documents\n")
    
    # Show all documents
    print("=" * 60)
    print("📚 DOCUMENTS IN DATABASE")
    print("=" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"{i:2d}. {doc}")
    print()
    
    # Perform semantic searches
    print("=" * 60)
    print("🔍 SEMANTIC SEARCH EXAMPLES")
    print("=" * 60)
    print()
    
    # Example 1: Search about neural networks
    query1 = "How do neural networks work?"
    print(f"Query 1: '{query1}'")
    print("-" * 60)
    
    results1 = collection.query(
        query_texts=[query1],
        n_results=3
    )
    
    print("Top 3 Results:")
    for i, (doc, distance) in enumerate(zip(results1['documents'][0], results1['distances'][0]), 1):
        similarity = 1 - distance  # Convert distance to similarity
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"\n{i}. [{bar}] Similarity: {similarity:.3f}")
        print(f"   {doc}")
    
    print("\n" + "=" * 60)
    
    # Example 2: Search about learning types
    query2 = "What are different types of machine learning?"
    print(f"\nQuery 2: '{query2}'")
    print("-" * 60)
    
    results2 = collection.query(
        query_texts=[query2],
        n_results=3
    )
    
    print("Top 3 Results:")
    for i, (doc, distance) in enumerate(zip(results2['documents'][0], results2['distances'][0]), 1):
        similarity = 1 - distance
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"\n{i}. [{bar}] Similarity: {similarity:.3f}")
        print(f"   {doc}")
    
    print("\n" + "=" * 60)
    
    # Example 3: Unrelated query
    query3 = "What's the best food?"
    print(f"\nQuery 3: '{query3}'")
    print("-" * 60)
    
    results3 = collection.query(
        query_texts=[query3],
        n_results=3
    )
    
    print("Top 3 Results:")
    for i, (doc, distance) in enumerate(zip(results3['documents'][0], results3['distances'][0]), 1):
        similarity = 1 - distance
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"\n{i}. [{bar}] Similarity: {similarity:.3f}")
        print(f"   {doc}")
    
    print("\n" + "=" * 60)
    print("💡 KEY INSIGHTS")
    print("=" * 60)
    print()
    print("✅ Semantic search finds MEANING, not just keywords!")
    print("   - 'How do neural networks work?' → Found neural network docs")
    print("   - 'Types of machine learning' → Found supervised/unsupervised docs")
    print()
    print("✅ Notice: No exact keyword matches needed!")
    print("   - Query used 'work', docs used 'process', 'learn', etc.")
    print()
    print("✅ This is why RAG is powerful:")
    print("   - Understands what you're asking about")
    print("   - Finds relevant info even with different wording")
    print("   - Returns the most helpful documents")
    print()

if __name__ == "__main__":
    main()
