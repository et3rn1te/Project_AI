# RAG Learning Examples

This folder contains interactive demos to help you understand how RAG works!

## 📚 Learning Order

Run these demos in order to build your understanding:

### 1️⃣ Embeddings Demo
**File**: `01_embeddings_demo.py`

**What you'll learn**:
- How text is converted to numbers (embeddings)
- How similarity is measured
- Why similar sentences have similar embeddings

**Run it**:
```bash
.\venv\Scripts\activate
python examples/01_embeddings_demo.py
```

**Expected output**: You'll see how AI-related sentences cluster together with high similarity scores!

---

### 2️⃣ Vector Search Demo
**File**: `02_vector_search_demo.py`

**What you'll learn**:
- How ChromaDB stores and searches documents
- Semantic search vs keyword search
- How to retrieve relevant documents

**Run it**:
```bash
.\venv\Scripts\activate
python examples/02_vector_search_demo.py
```

**Expected output**: You'll see how ChromaDB finds relevant documents even when you use different words!

---

### 3️⃣ Complete RAG Workflow Demo
**File**: `03_rag_workflow_demo.py`

**What you'll learn**:
- The entire RAG process from start to finish
- How retrieval and generation work together
- How context improves LLM answers

**Requirements**: Ollama must be running!

**Run it**:
```bash
# Make sure Ollama is running (check with: ollama ps)
.\venv\Scripts\activate
python examples/03_rag_workflow_demo.py
```

**Expected output**: You'll see a complete RAG workflow answering a question about supervised learning!

---

## 🎯 What These Demos Teach

After running all three demos, you'll understand:

✅ **Embeddings**: How text becomes searchable vectors  
✅ **Vector Databases**: How ChromaDB finds relevant information  
✅ **RAG Workflow**: How all the pieces work together  
✅ **Semantic Search**: Why RAG is smarter than keyword search  

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
Make sure your virtual environment is activated:
```bash
.\venv\Scripts\activate
```

### "Ollama connection error" (Demo 3 only)
Make sure Ollama is running:
```bash
ollama ps
# If nothing is running, Ollama is ready
# If you get an error, start Ollama from the Start menu
```

### Slow performance
- First run downloads the embedding model (~80MB)
- Demo 3 takes 10-30 seconds to generate answers (normal!)

---

## 💡 Next Steps

After understanding these concepts:
1. Review the code in each demo
2. Try modifying the example sentences/documents
3. Experiment with different questions
4. Move on to Phase 3: Building the actual RAG system!

---

## 🎓 Learning Tips

- **Run each demo multiple times** with different inputs
- **Read the code** - it's heavily commented to explain what's happening
- **Experiment** - change the sentences, questions, and see what happens
- **Ask questions** - if something doesn't make sense, ask!

Happy learning! 🚀
