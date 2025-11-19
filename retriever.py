import os
from langchain_chroma import Chroma
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

def load_vector_store(
    embeddings: str,
) -> Chroma:
    """
    embeddings: supports following embeddings to query from: static, dense, minilm
    """
    embedding_options = {
        "static": Model2vecEmbeddings("minishlab/potion-base-8M"),
        "dense": HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5"),
        "minilm": HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    }

    if embeddings in embedding_options:
        if os.path.exists("./data/chromadb"):
            return Chroma(
                persist_directory="./data/chromadb",
                embedding_function=embedding_options[embeddings],
                collection_name=f"{embeddings}_collection"
            )
        else:
            raise FileNotFoundError("Directory does not exist. Run dataloader.py to generate chromadb embeddings with persistence.")
        
    else:
        raise ValueError(f"Unsupported embeddings type: {embeddings}. Supported types are: {list(embedding_options.keys())}.")


def hybrid_retrieval(
    documents: list[Document],
    embeddings: Model2vecEmbeddings | HuggingFaceEmbeddings,
    collection_name: str
) -> EnsembleRetriever:
    # TODO: implement retriever from persistence db rather than documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    vector_store_retriever = load_vector_store(embeddings, collection_name).as_retriever()
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store_retriever],
        weights=[0.5, 0.5]
    )
