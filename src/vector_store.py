"""
Vector Store Module

This module handles storing and retrieving document embeddings using ChromaDB.

Simple Explanation:
This is your "smart library system" that:
1. Stores all your document chunks with their embeddings
2. Finds the most relevant chunks when you ask a question
3. Remembers where each chunk came from

Think of it as a librarian with a photographic memory who can instantly
find the most relevant books for any question!
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from src.document_loader import Document
from src.embeddings import EmbeddingGenerator


class VectorStore:
    """
    Manages document storage and retrieval using ChromaDB.
    
    Simple Explanation:
    This is your document database that understands meaning, not just keywords!
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for your document collection
            persist_directory: Where to save the database
            embedding_model: Which embedding model to use
            
        Simple Explanation:
        - collection_name: Like naming your library (e.g., "AI Learning Docs")
        - persist_directory: Where to save everything (so it's not lost when you close)
        - embedding_model: The "translator" that converts text to vectors
        """
        print(f"🔧 Initializing Vector Store...")
        print(f"   Collection: {collection_name}")
        print(f"   Storage: {persist_directory}")
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "RAG document collection"}
        )
        
        print(f"✅ Vector Store ready!")
        print(f"   Current documents: {self.collection.count()}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Simple Explanation:
        Takes your document chunks and:
        1. Converts them to embeddings (automatically by ChromaDB)
        2. Stores them with their metadata
        3. Makes them searchable!
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            print("⚠️  No documents to add!")
            return
        
        print(f"\n📥 Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Create unique ID
            doc_id = f"doc_{self.collection.count() + i}_{hash(doc.content) % 10000}"
            
            ids.append(doc_id)
            texts.append(doc.content)
            metadatas.append(doc.metadata)
        
        # Add to collection (ChromaDB will generate embeddings automatically)
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(documents)} documents!")
        print(f"   Total documents in store: {self.collection.count()}")
    
    def search(
        self,
        query: str,
        n_results: int = 4,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Search for relevant documents.
        
        Simple Explanation:
        When you ask a question:
        1. Converts your question to an embedding
        2. Finds documents with similar embeddings
        3. Returns the most relevant ones!
        
        Args:
            query: Your question or search text
            n_results: How many results to return (default: 4)
            filter_metadata: Optional filters (e.g., only PDFs)
            
        Returns:
            Dictionary with documents, distances, and metadata
        """
        if not query or not query.strip():
            print("⚠️  Empty query!")
            return {"documents": [], "distances": [], "metadatas": []}
        
        print(f"\n🔍 Searching for: '{query}'")
        print(f"   Retrieving top {n_results} results...")
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata  # Optional filtering
        )
        
        # Extract results
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        print(f"✅ Found {len(documents)} relevant documents")
        
        return {
            "documents": documents,
            "distances": distances,
            "metadatas": metadatas
        }
    
    def get_formatted_results(self, search_results: Dict) -> List[Dict]:
        """
        Format search results in a nice, readable way.
        
        Simple Explanation:
        Takes the raw search results and makes them pretty and easy to use!
        
        Returns:
            List of dictionaries with document info and similarity scores
        """
        formatted = []
        
        documents = search_results.get("documents", [])
        distances = search_results.get("distances", [])
        metadatas = search_results.get("metadatas", [])
        
        for doc, distance, metadata in zip(documents, distances, metadatas):
            # Convert distance to similarity (0 to 1)
            similarity = 1 - distance
            
            formatted.append({
                "content": doc,
                "similarity": similarity,
                "metadata": metadata,
                "source": metadata.get("filename", "Unknown"),
                "page": metadata.get("page", None)
            })
        
        return formatted
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        
        Simple Explanation:
        Empties the library - removes all documents.
        Use this to start fresh!
        """
        print(f"🗑️  Clearing collection '{self.collection.name}'...")
        
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        
        print("✅ Collection cleared!")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("📚 VECTOR STORE TEST")
    print("=" * 60)
    print()
    
    # Initialize vector store
    store = VectorStore(collection_name="test_collection")
    
    # Create sample documents
    from src.document_loader import Document
    
    sample_docs = [
        Document(
            content="Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes.",
            metadata={"filename": "neural_networks.txt", "topic": "AI"}
        ),
        Document(
            content="Supervised learning is a machine learning approach where models are trained on labeled data.",
            metadata={"filename": "ml_basics.txt", "topic": "ML"}
        ),
        Document(
            content="Deep learning uses multiple layers to learn hierarchical representations of data.",
            metadata={"filename": "deep_learning.txt", "topic": "DL"}
        ),
        Document(
            content="Python is a popular programming language widely used in data science and machine learning.",
            metadata={"filename": "python_intro.txt", "topic": "Programming"}
        )
    ]
    
    # Add documents
    store.add_documents(sample_docs)
    
    # Test search
    print("\n" + "=" * 60)
    print("TEST SEARCH")
    print("=" * 60)
    
    query = "How do neural networks work?"
    results = store.search(query, n_results=3)
    
    # Format and display results
    formatted = store.get_formatted_results(results)
    
    print(f"\nResults for: '{query}'")
    print("-" * 60)
    
    for i, result in enumerate(formatted, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Source: {result['source']}")
        print(f"   Content: {result['content'][:100]}...")
    
    # Show stats
    print("\n" + "=" * 60)
    print("VECTOR STORE STATS")
    print("=" * 60)
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Vector Store test complete!")
