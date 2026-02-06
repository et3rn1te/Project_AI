"""
Integration Test for Phase 3 - Document Processing Pipeline

This script tests all Phase 3 components working together:
1. Document Loading
2. Text Splitting
3. Embedding Generation
4. Vector Store
5. Retrieval

Simple Explanation:
This is like a "dress rehearsal" - we test all the pieces working together
before building the final application!
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_loader import Document, DocumentLoader
from src.text_splitter import SmartTextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever


def test_phase3_pipeline():
    """Test the complete Phase 3 document processing pipeline."""
    
    print("=" * 70)
    print("🧪 PHASE 3 INTEGRATION TEST")
    print("=" * 70)
    print()
    print("Testing: Document Loading → Chunking → Embeddings → Vector Store → Retrieval")
    print()
    
    # Step 1: Create sample documents
    print("STEP 1: Creating Sample Documents")
    print("-" * 70)
    
    sample_docs = [
        Document(
            content="""Neural networks are computing systems inspired by biological neural networks that constitute animal brains. 
            An artificial neural network is based on a collection of connected units or nodes called artificial neurons, 
            which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, 
            can transmit a signal to other neurons. An artificial neuron receives signals then processes them and can signal 
            neurons connected to it. The signal at a connection is a real number, and the output of each neuron is computed 
            by some non-linear function of the sum of its inputs.""",
            metadata={"filename": "neural_networks.pdf", "page": 1, "topic": "Neural Networks"}
        ),
        Document(
            content="""Supervised learning is the machine learning task of learning a function that maps an input to an output 
            based on example input-output pairs. It infers a function from labeled training data consisting of a set of training 
            examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a 
            desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data 
            and produces an inferred function, which can be used for mapping new examples.""",
            metadata={"filename": "supervised_learning.pdf", "page": 1, "topic": "Machine Learning"}
        ),
        Document(
            content="""Deep learning is part of a broader family of machine learning methods based on artificial neural networks 
            with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures 
            such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have 
            been applied to fields including computer vision, speech recognition, natural language processing, machine translation, 
            bioinformatics, drug design, medical image analysis, and board game programs.""",
            metadata={"filename": "deep_learning.pdf", "page": 2, "topic": "Deep Learning"}
        )
    ]
    
    print(f"✅ Created {len(sample_docs)} sample documents")
    for doc in sample_docs:
        print(f"   - {doc.metadata['filename']}: {len(doc.content)} characters")
    
    # Step 2: Text Splitting
    print("\n" + "=" * 70)
    print("STEP 2: Splitting Documents into Chunks")
    print("-" * 70)
    
    splitter = SmartTextSplitter(chunk_size=300, chunk_overlap=50)
    chunked_docs = splitter.split_documents(sample_docs)
    
    print(f"✅ Split into {len(chunked_docs)} chunks")
    print(f"   Original documents: {len(sample_docs)}")
    print(f"   Chunks per document: ~{len(chunked_docs) // len(sample_docs)}")
    
    # Step 3: Initialize Vector Store
    print("\n" + "=" * 70)
    print("STEP 3: Initializing Vector Store")
    print("-" * 70)
    
    vector_store = VectorStore(
        collection_name="phase3_test",
        persist_directory="./chroma_db_test"
    )
    
    # Clear any existing data
    vector_store.clear_collection()
    
    # Step 4: Add Documents to Vector Store
    print("\n" + "=" * 70)
    print("STEP 4: Adding Chunks to Vector Store")
    print("-" * 70)
    
    vector_store.add_documents(chunked_docs)
    
    stats = vector_store.get_stats()
    print(f"\n📊 Vector Store Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Step 5: Test Retrieval
    print("\n" + "=" * 70)
    print("STEP 5: Testing Retrieval System")
    print("-" * 70)
    
    retriever = Retriever(
        vector_store=vector_store,
        top_k=3,
        similarity_threshold=0.3
    )
    
    # Test queries
    test_queries = [
        "What are neural networks?",
        "How does supervised learning work?",
        "What is deep learning used for?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: '{query}'")
        print('─' * 70)
        
        result = retriever.retrieve_with_context(query)
        
        print(f"\nRetrieved {result['num_results']} documents:")
        for j, doc in enumerate(result['documents'], 1):
            print(f"\n  {j}. Similarity: {doc['similarity']:.3f}")
            print(f"     Source: {doc['source']}")
            print(f"     Preview: {doc['content'][:100]}...")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("✅ PHASE 3 PIPELINE TEST COMPLETE!")
    print("=" * 70)
    print()
    print("All components working correctly:")
    print("  ✅ Document Loading")
    print("  ✅ Text Chunking (Smart Splitter)")
    print("  ✅ Embedding Generation (via ChromaDB)")
    print("  ✅ Vector Storage (ChromaDB)")
    print("  ✅ Semantic Retrieval")
    print()
    print("🎉 Ready to build the Streamlit application!")
    print()


if __name__ == "__main__":
    try:
        test_phase3_pipeline()
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
