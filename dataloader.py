import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_options = {
    "static": Model2vecEmbeddings("minishlab/potion-base-8M"),
    "dense": HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5"),
    "minilm": HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
}

print("Loading dataset")
collection_split = pd.read_json("./data/collection.jsonl", lines=True)

# TODO: Remove row data with duplicate text value
unique_collection_split = collection_split.copy().drop_duplicates(subset=['text'], keep='first')

collection_loader = DataFrameLoader(unique_collection_split, page_content_column="text")
collection_docs = collection_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
print("Splitting loaded documents!")
collection_text_splitter = text_splitter.split_documents(collection_docs)

print("Generating vector embeddings!")
for each in embedding_options.keys():
    print(f"Generating vector embeddings for {each}_collections")
    Chroma.from_documents(
        documents=collection_text_splitter,
        embedding=embedding_options[each],
        persist_directory="./data/chromadb",
        collection_name=f"{each}_collection"
    )
