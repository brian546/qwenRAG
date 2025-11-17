# QwenRAG

This repository implements a Retrieval-Augmented Generation (RAG) system using the Qwen-2.5-instruct model. The system integrates various vector store for retrieval and provides a StreamLit-based chatbot interface for interactive question answering based on HQ-small dataset.

## Set up environment
```bash
conda env create -n rag -f environment.yml
conda activate rag
```

## Run Chatbot
Execute the following command to start a Streamlit-based chatbot application. After running it, Streamlit will display a local URL (usually http://localhost:8501). Open that URL in your web browser to access and interact with the chatbot app.

```bash
streamlit run chatbot.py
```