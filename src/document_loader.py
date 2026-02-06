"""
Document Loader Module

This module handles loading documents from various file formats:
- PDF files
- Text files (.txt)
- Markdown files (.md)

Simple Explanation:
Think of this as a "universal reader" that can open different types of files
and extract the text content from them.
"""

import os
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


class Document:
    """
    A simple container for document content and metadata.
    
    Think of this as a "note card" that holds:
    - The actual text content
    - Information about where it came from (filename, page number, etc.)
    """
    
    def __init__(self, content: str, metadata: Dict[str, any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentLoader:
    """
    Loads documents from various file formats.
    
    Simple Explanation:
    Like a librarian who can read different types of books (PDF, text, markdown)
    and tell you what's written in them.
    """
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt', '.md'}
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file and return Document objects.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects (one per page for PDFs, one for text files)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}. Supported: {self.supported_extensions}")
        
        # Route to appropriate loader based on file type
        if extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension in {'.txt', '.md'}:
            return self._load_text(file_path)
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a PDF file and extract text from each page.
        
        Simple Explanation:
        Opens a PDF and reads each page, creating a separate Document for each page.
        This way we know which page information came from!
        """
        documents = []
        
        try:
            reader = PdfReader(str(file_path))
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                # Only create document if page has text
                if text.strip():
                    doc = Document(
                        content=text,
                        metadata={
                            'source': str(file_path),
                            'filename': file_path.name,
                            'page': page_num,
                            'total_pages': len(reader.pages),
                            'file_type': 'pdf'
                        }
                    )
                    documents.append(doc)
            
            print(f"✅ Loaded {len(documents)} pages from {file_path.name}")
            
        except Exception as e:
            print(f"❌ Error loading PDF {file_path.name}: {str(e)}")
            raise
        
        return documents
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """
        Load a text or markdown file.
        
        Simple Explanation:
        Opens a text file and reads all the content at once.
        Creates one Document for the entire file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if text.strip():
                doc = Document(
                    content=text,
                    metadata={
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_type': file_path.suffix[1:]  # Remove the dot
                    }
                )
                
                print(f"✅ Loaded {file_path.name} ({len(text)} characters)")
                return [doc]
            else:
                print(f"⚠️  File {file_path.name} is empty")
                return []
                
        except Exception as e:
            print(f"❌ Error loading text file {file_path.name}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Simple Explanation:
        Like reading all the books in a library folder!
        Goes through each file and loads the ones we can read.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all Document objects from all files
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        all_documents = []
        
        # Find all supported files
        for extension in self.supported_extensions:
            files = list(directory_path.glob(f"*{extension}"))
            
            for file_path in files:
                try:
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"⚠️  Skipping {file_path.name}: {str(e)}")
        
        print(f"\n📚 Total: Loaded {len(all_documents)} documents from {directory_path}")
        return all_documents


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("📄 DOCUMENT LOADER TEST")
    print("=" * 60)
    print()
    
    loader = DocumentLoader()
    
    # Test loading from data directory
    data_dir = Path("data")
    
    if data_dir.exists():
        print(f"Loading documents from: {data_dir}")
        print("-" * 60)
        
        try:
            documents = loader.load_directory(data_dir)
            
            if documents:
                print("\n" + "=" * 60)
                print("📊 LOADED DOCUMENTS SUMMARY")
                print("=" * 60)
                
                for i, doc in enumerate(documents[:5], 1):  # Show first 5
                    print(f"\n{i}. {doc.metadata['filename']}")
                    print(f"   Type: {doc.metadata['file_type']}")
                    if 'page' in doc.metadata:
                        print(f"   Page: {doc.metadata['page']}/{doc.metadata['total_pages']}")
                    print(f"   Content preview: {doc.content[:100]}...")
                
                if len(documents) > 5:
                    print(f"\n... and {len(documents) - 5} more documents")
            else:
                print("\n⚠️  No documents found!")
                print("💡 Add some PDF, TXT, or MD files to the 'data' folder to test!")
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    else:
        print("⚠️  'data' directory not found!")
        print("💡 Create it and add some documents to test the loader!")
