# QwenRAG

This repository implements a Retrieval-Augmented Generation (RAG) system using the Qwen-2.5-instruct model from ollama in Linux. The system integrates various vector store for retrieval and provides a StreamLit-based chatbot interface for interactive question answering based on HQ-small dataset.

## Project Structure

- `batch_generate.py`: Batch-generation utilities and scripts for producing multiple queries or answers.
- `chatbot.py`: Chat interface / CLI or Streamlit entrypoint that interacts with the RAG pipeline.
- `dataloader.py`: Loads `.jsonl` datasets into memory and preprocesses text for indexing.
- `environment.yml`: Conda environment specification with required packages.

- `rag.py`: Core RAG orchestration — ties retriever, generator, and prompt handling together.
- `ragtype.py`: Type definitions for messages and RAG state (see details below).
- `retriever.py`: Retriever implementation — builds/queries the vector DB (Chroma/SQLite) and returns docs.
- `eval/`:
    - `eval_hotpotqa.py`: Evaluation script for HotpotQA-style QA using retrieval + generation metrics.
    - `eval_retrieval.py`: Evaluation script focused on retrieval metrics (recall@k, precision, etc.).
- `data/`:
  - `collection.jsonl`: Collection used to build the vector DB.
  - `train.jsonl`, `validation.jsonl`, `test.jsonl`: Datasets for training/eval.
  - `chromadb/`:
    - `chroma.sqlite3`: Chroma DB SQLite file (storage).
    - `<uuid>/` directories: internal Chroma data shards.

## Set up environment
To set up the required environment, first intall ollama from [ollama.com](https://ollama.com/). After that, install qwen2.5-instruct model by running:
```bash
ollama pull qwen2.5:7b-instruct
```

Then, create and activate the conda environment using the provided `environment.yml` file:
```bash
conda env create -n rag -f environment.yml
conda activate rag
```
## Build Vector Store
Run the following command to build the vector store from the HQ-small dataset. This process involves loading the dataset, processing it, and storing it in a vector database for efficient retrieval.
```bash
python dataloader.py
```

## Run Chatbot
Execute the following command to start a Streamlit-based chatbot application. After running it, Streamlit will display a local URL (usually http://localhost:8501). Open that URL in your web browser to access and interact with the chatbot app.

```bash
streamlit run chatbot.py
```

## Run retrieval and generation evaluation
To evaluate the retrieval and generation performance of the RAG system, you can use the provided evaluation scripts. For retrieval evaluation, run:
```bash
python eval/eval_retrieval.py --gold data/validation.jsonl --pred PREDICTED_FILE
``` 
For generation evaluation, run:
```bash
python eval/eval_hotpotqa.py --gold data/validation.jsonl --pred PREDICTED_FILE
``` 

