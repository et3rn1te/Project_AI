# RAG Learning System

A Retrieval-Augmented Generation (RAG) system built with free, open-source tools to help you learn AI engineering concepts.

## 🚀 Quick Start

1. **Activate Virtual Environment**
   ```bash
   .\venv\Scripts\activate
   ```

2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama** (if not running)
   ```bash
   ollama serve
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Project_AI/
├── venv/                  # Virtual environment
├── data/                  # Your learning materials
│   ├── pdfs/             # PDF documents
│   ├── markdown/         # Markdown files
│   └── text/             # Text files
├── chroma_db/            # Vector database storage
├── src/                  # Source code modules
├── app.py                # Main Streamlit application
├── config.py             # Configuration settings
├── .env                  # Environment variables
└── requirements.txt      # Python dependencies
```

## 🛠️ Technology Stack

- **LLM**: Ollama (Mistral 7B / Llama 3)
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Framework**: LangChain
- **Interface**: Streamlit
- **Language**: Python 3.13

## ✅ Current Setup Status

- ✅ Python 3.13.9 installed
- ✅ Virtual environment created
- ✅ Ollama 0.15.4 installed
- ✅ Models: mistral:7b, llama3:latest
- ✅ All dependencies installed
- ✅ Project structure created

## 📚 Learning Resources

See [implementation_plan.md](C:\Users\Nguyen Duc Tai\.gemini\antigravity\brain\64789466-db0c-41b3-bb48-e5e499c7bd4c\implementation_plan.md) for detailed explanations and step-by-step guide.

## 🎯 Next Steps

1. Create source code modules in `src/`
2. Build the Streamlit interface in `app.py`
3. Add your learning materials to `data/`
4. Start asking questions!
