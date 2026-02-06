"""
Prompt Templates Module

This module contains all the prompt templates for the RAG system.

Simple Explanation:
These are the "instruction scripts" we give to the AI.
Good prompts = Good answers!

Think of these as recipe cards - they tell the AI exactly how to
use the context and answer your question.
"""


class PromptTemplates:
    """
    Collection of prompt templates for different use cases.
    
    Simple Explanation:
    Different types of questions need different instructions!
    """
    
    @staticmethod
    def rag_qa_prompt(context: str, question: str) -> str:
        """
        Standard RAG Q&A prompt template.
        
        Simple Explanation:
        This is the main template for answering questions using your documents.
        
        Args:
            context: Retrieved document chunks
            question: User's question
            
        Returns:
            Complete prompt ready for the LLM
        """
        prompt = f"""You are a helpful AI tutor teaching AI engineering concepts.

Context from learning materials:
{context}

Student Question: {question}

Instructions:
- Answer based ONLY on the provided context above
- Explain concepts simply and clearly, like teaching a beginner
- Use examples from the context when helpful
- If the context doesn't contain enough information to answer fully, say so honestly
- Be concise but thorough

Answer:"""
        
        return prompt
    
    @staticmethod
    def rag_qa_with_sources_prompt(context: str, question: str) -> str:
        """
        RAG Q&A prompt that asks for source citations.
        
        Simple Explanation:
        Same as above, but also asks the AI to mention which documents it used.
        """
        prompt = f"""You are a helpful AI tutor teaching AI engineering concepts.

Context from learning materials:
{context}

Student Question: {question}

Instructions:
- Answer based ONLY on the provided context above
- Explain concepts simply and clearly, like teaching a beginner
- Use examples from the context when helpful
- Mention which documents you're referencing (e.g., "According to Document 1...")
- If the context doesn't contain enough information to answer fully, say so honestly
- Be concise but thorough

Answer:"""
        
        return prompt
    
    @staticmethod
    def conversational_rag_prompt(context: str, chat_history: str, question: str) -> str:
        """
        RAG prompt with conversation history.
        
        Simple Explanation:
        This remembers previous questions and answers, so you can ask follow-up
        questions like "Can you explain that more?" and it knows what "that" means!
        
        Args:
            context: Retrieved document chunks
            chat_history: Previous Q&A pairs
            question: Current question
        """
        prompt = f"""You are a helpful AI tutor teaching AI engineering concepts.

Previous Conversation:
{chat_history}

Context from learning materials:
{context}

Current Question: {question}

Instructions:
- Consider the conversation history when answering
- Answer based on the provided context
- Explain concepts simply and clearly
- If the question refers to something from the conversation history, acknowledge it
- Be concise but thorough

Answer:"""
        
        return prompt
    
    @staticmethod
    def summarization_prompt(text: str) -> str:
        """
        Prompt for summarizing documents.
        
        Simple Explanation:
        Asks the AI to create a short summary of a long document.
        """
        prompt = f"""Please provide a concise summary of the following text.
Focus on the main points and key concepts.

Text:
{text}

Summary:"""
        
        return prompt
    
    @staticmethod
    def no_context_prompt(question: str) -> str:
        """
        Fallback prompt when no relevant context is found.
        
        Simple Explanation:
        If we can't find relevant documents, we still try to answer
        but tell the user it's based on general knowledge, not their docs.
        """
        prompt = f"""You are a helpful AI tutor teaching AI engineering concepts.

Note: No relevant context was found in the learning materials for this question.

Question: {question}

Instructions:
- Answer based on your general knowledge
- Clearly state that this answer is not from the provided learning materials
- Suggest that the user might want to add relevant materials to their knowledge base
- Keep the answer concise and helpful

Answer:"""
        
        return prompt


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("📝 PROMPT TEMPLATES TEST")
    print("=" * 60)
    print()
    
    # Sample context and question
    sample_context = """[Document 1 - neural_networks.pdf (Page 1)]
Neural networks are computing systems inspired by biological neural networks.
They consist of layers of interconnected nodes that process information.

[Document 2 - deep_learning.pdf (Page 3)]
Deep learning uses multiple layers to learn hierarchical representations.
Each layer learns increasingly complex features from the data."""
    
    sample_question = "What are neural networks?"
    
    # Test standard RAG prompt
    print("TEST 1: Standard RAG Q&A Prompt")
    print("=" * 60)
    
    prompt1 = PromptTemplates.rag_qa_prompt(sample_context, sample_question)
    print(prompt1)
    
    print("\n\n" + "=" * 60)
    print("TEST 2: RAG Q&A with Sources Prompt")
    print("=" * 60)
    
    prompt2 = PromptTemplates.rag_qa_with_sources_prompt(sample_context, sample_question)
    print(prompt2)
    
    print("\n\n" + "=" * 60)
    print("TEST 3: Conversational RAG Prompt")
    print("=" * 60)
    
    chat_history = """Q: What is machine learning?
A: Machine learning is a subset of AI that focuses on algorithms that learn from data.

Q: Can you give an example?
A: Sure! Email spam filters learn to identify spam by analyzing many examples of spam and non-spam emails."""
    
    follow_up = "How does that relate to neural networks?"
    
    prompt3 = PromptTemplates.conversational_rag_prompt(sample_context, chat_history, follow_up)
    print(prompt3)
    
    print("\n\n" + "=" * 60)
    print("✅ Prompt templates test complete!")
    print("=" * 60)
