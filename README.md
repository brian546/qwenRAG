# QwenRAG

This repository implements a Retrieval-Augmented Generation (RAG) system using the Qwen-2.5-instruct model. The system integrates various vector store for retrieval and provides a StreamLit-based chatbot interface for interactive question answering based on HQ-small dataset.

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

## Run Chatbot
Execute the following command to start a Streamlit-based chatbot application. After running it, Streamlit will display a local URL (usually http://localhost:8501). Open that URL in your web browser to access and interact with the chatbot app.

```bash
streamlit run chatbot.py
```