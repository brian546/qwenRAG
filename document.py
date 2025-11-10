from dataclasses import dataclass
from typing import List, Dict, Any, Deque

@dataclass
class Document:
    """Class to represent a document chunk."""
    text: str
    metadata: Dict[str, Any]

# @dataclass
# class Prompt:
#     query: str
#     history: Deque[Dict[str,str]]
#     context: List[Document]


def split_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def load_documents(directory: str) -> List[Document]:
    """Load documents from a directory."""
    documents = []
    
    # This is a placeholder - implement actual document loading based on your data
    # For demonstration, we'll create some sample documents
    sample_texts = [
        "Barack Obama was born in Hawaii. He served as the 44th president of the United States.",
        "Michelle Obama was born in Chicago, Illinois. She served as First Lady of the United States.",
        "Barack and Michelle Obama met in Chicago and got married in 1992.",
        "Kathy lives in Lai King"
    ]
    
    for i, text in enumerate(sample_texts):
        chunks = split_text(text)
        for chunk in chunks:
            doc = Document(
                text=chunk,
                metadata={'source_id': i, 'source_text': text}
            )
            documents.append(doc)
    
    return documents