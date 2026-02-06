"""
Complete RAG Workflow Demo

This script demonstrates the ENTIRE RAG process:
1. Store documents in vector database
2. User asks a question
3. Find relevant documents
4. Send to LLM with context
5. Get an answer

This is a mini version of what we'll build!
"""

import chromadb
from chromadb.utils import embedding_functions
import requests
import json

def query_ollama(prompt, model="mistral:7b"):
    """Send a prompt to Ollama and get a response"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 300
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure Ollama is running: 'ollama serve'"

def main():
    print("=" * 70)
    print("🤖 COMPLETE RAG WORKFLOW DEMO")
    print("=" * 70)
    print()
    print("This demo shows the ENTIRE RAG process step-by-step!")
    print()
    
    # Step 1: Set up vector database
    print("STEP 1: Setting up Vector Database")
    print("-" * 70)
    
    client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.create_collection(
        name="ai_knowledge",
        embedding_function=embedding_function
    )
    
    # AI learning documents
    documents = [
        "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process and transmit information.",
        
        "Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs.",
        
        "Unsupervised learning is machine learning where the algorithm finds patterns in data without labeled responses. Common techniques include clustering and dimensionality reduction.",
        
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers. Each layer learns increasingly complex features from the data.",
        
        "Embeddings are vector representations of data that capture semantic meaning. Similar items have similar embeddings, enabling semantic search and similarity comparisons.",
        
        "Transfer learning involves taking a pre-trained model and fine-tuning it for a new task. This saves time and computational resources compared to training from scratch.",
        
        "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning. It iteratively adjusts model parameters to find the optimal values.",
        
        "Overfitting occurs when a model learns the training data too well, including noise and outliers. This results in poor performance on new, unseen data."
    ]
    
    print(f"📚 Adding {len(documents)} AI learning documents...")
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print("✅ Documents stored in vector database\n")
    
    # Step 2: User asks a question
    print("STEP 2: User Asks a Question")
    print("-" * 70)
    
    user_question = "What is supervised learning and how does it work?"
    print(f"❓ Question: '{user_question}'\n")
    
    # Step 3: Retrieve relevant documents
    print("STEP 3: Searching for Relevant Documents")
    print("-" * 70)
    
    print("🔍 Converting question to embedding and searching...")
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )
    
    retrieved_docs = results['documents'][0]
    distances = results['distances'][0]
    
    print(f"✅ Found {len(retrieved_docs)} relevant documents:\n")
    
    for i, (doc, distance) in enumerate(zip(retrieved_docs, distances), 1):
        similarity = 1 - distance
        print(f"{i}. [Similarity: {similarity:.3f}]")
        print(f"   {doc[:100]}...")
        print()
    
    # Step 4: Create prompt with context
    print("STEP 4: Creating Prompt for LLM")
    print("-" * 70)
    
    # Build context from retrieved documents
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
    # Create the RAG prompt
    prompt = f"""You are a helpful AI tutor teaching AI engineering concepts.

Context from learning materials:
{context}

Student Question: {user_question}

Instructions:
- Answer based on the provided context
- Explain concepts simply and clearly
- Use examples if helpful
- If the context doesn't fully answer the question, say so

Answer:"""
    
    print("📝 Prompt created with context from retrieved documents")
    print(f"   Prompt length: {len(prompt)} characters\n")
    
    # Step 5: Get answer from LLM
    print("STEP 5: Generating Answer with LLM")
    print("-" * 70)
    
    print("🤖 Sending to Ollama (mistral:7b)...")
    print("⏳ Generating answer (this may take 10-30 seconds)...\n")
    
    answer = query_ollama(prompt)
    
    print("=" * 70)
    print("✨ FINAL ANSWER")
    print("=" * 70)
    print()
    print(answer)
    print()
    
    # Summary
    print("=" * 70)
    print("📊 RAG WORKFLOW SUMMARY")
    print("=" * 70)
    print()
    print("✅ Step 1: Stored documents in ChromaDB")
    print("✅ Step 2: Received user question")
    print("✅ Step 3: Retrieved 3 most relevant documents")
    print("✅ Step 4: Created prompt with context")
    print("✅ Step 5: Generated answer using Ollama")
    print()
    print("💡 This is exactly what your RAG system will do!")
    print()
    print("Key Benefits:")
    print("  • Answer is based on YOUR documents (not just LLM knowledge)")
    print("  • Retrieval finds relevant info automatically")
    print("  • LLM explains it in a helpful way")
    print("  • Works with any documents you provide")
    print()

if __name__ == "__main__":
    main()
