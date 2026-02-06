"""
Retriever Module

This module combines the vector store with smart retrieval logic.

Simple Explanation:
This is the "smart search assistant" that:
1. Takes your question
2. Finds the most relevant document chunks
3. Formats them nicely for the LLM

Think of it as a research assistant who finds exactly the right pages
from your textbooks to answer your question!
"""

from typing import List, Dict
from src.vector_store import VectorStore


class Retriever:
    """
    Handles document retrieval with smart filtering and ranking.
    
    Simple Explanation:
    This finds the best documents to answer your question!
    """
    
    def __init__(self, vector_store: VectorStore, top_k: int = 4, similarity_threshold: float = 0.5):
        """
        Initialize the retriever.
        
        Args:
            vector_store: The VectorStore instance to search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Simple Explanation:
        - top_k: How many chunks to retrieve (usually 3-5)
        - similarity_threshold: Only return chunks above this similarity
                               (0.5 = 50% similar or more)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Simple Explanation:
        Searches for documents related to your question and returns
        only the good matches (above the similarity threshold).
        
        Args:
            query: Your question or search text
            
        Returns:
            List of relevant documents with metadata
        """
        # Search vector store
        results = self.vector_store.search(query, n_results=self.top_k)
        
        # Format results
        formatted_results = self.vector_store.get_formatted_results(results)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in formatted_results
            if result['similarity'] >= self.similarity_threshold
        ]
        
        if not filtered_results:
            print(f"⚠️  No documents found above similarity threshold {self.similarity_threshold}")
        else:
            print(f"✅ Retrieved {len(filtered_results)} relevant documents")
        
        return filtered_results
    
    def get_context_string(self, retrieved_docs: List[Dict]) -> str:
        """
        Convert retrieved documents into a formatted context string.
        
        Simple Explanation:
        Takes the retrieved chunks and combines them into one nice text
        that we can give to the LLM as context.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Formatted string with all document contents
        """
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Create source citation
            source = doc['metadata'].get('filename', 'Unknown')
            page = doc['metadata'].get('page')
            
            source_info = f"{source}"
            if page:
                source_info += f" (Page {page})"
            
            # Add document content with source
            context_parts.append(
                f"[Document {i} - {source_info}]\n{doc['content']}"
            )
        
        return "\n\n".join(context_parts)
    
    def retrieve_with_context(self, query: str) -> Dict:
        """
        Retrieve documents and return both raw results and formatted context.
        
        Simple Explanation:
        One-stop function that:
        1. Searches for relevant docs
        2. Formats them for the LLM
        3. Returns everything you need!
        
        Returns:
            Dictionary with 'documents' and 'context_string'
        """
        documents = self.retrieve(query)
        context_string = self.get_context_string(documents)
        
        return {
            "documents": documents,
            "context": context_string,
            "num_results": len(documents)
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 RETRIEVER TEST")
    print("=" * 60)
    print()
    
    from src.vector_store import VectorStore
    from src.document_loader import Document
    
    # Initialize vector store
    store = VectorStore(collection_name="retriever_test")
    
    # Add sample documents
    sample_docs = [
        Document(
            content="Neural networks consist of layers of interconnected nodes that process information. Each layer transforms the input data progressively.",
            metadata={"filename": "neural_nets.pdf", "page": 1}
        ),
        Document(
            content="Supervised learning requires labeled training data. The model learns to map inputs to outputs based on examples.",
            metadata={"filename": "ml_basics.pdf", "page": 3}
        ),
        Document(
            content="Deep learning uses multiple layers to learn hierarchical representations. It excels at tasks like image recognition.",
            metadata={"filename": "deep_learning.pdf", "page": 5}
        )
    ]
    
    store.add_documents(sample_docs)
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=store,
        top_k=3,
        similarity_threshold=0.3
    )
    
    # Test retrieval
    print("\n" + "=" * 60)
    print("TEST: Retrieve with Context")
    print("=" * 60)
    
    query = "How do neural networks work?"
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    
    result = retriever.retrieve_with_context(query)
    
    print(f"\nRetrieved {result['num_results']} documents")
    print("\n" + "=" * 60)
    print("FORMATTED CONTEXT")
    print("=" * 60)
    print(result['context'])
    
    print("\n" + "=" * 60)
    print("✅ Retriever test complete!")
    print("=" * 60)
