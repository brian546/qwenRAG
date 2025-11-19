from rag import load_graph
import uuid
import json
from pathlib import Path
from tqdm import tqdm
import os

# ===================
#      Config
# ===================
DATA_FILE = Path("data/validation.jsonl")
RESULT_DIR = 'results'
OUTPUT_FILE = Path(f"{RESULT_DIR}/predictions.jsonl")
EMBEDDING_TYPE = "minilm"


# create result directory if not exists
os.makedirs(RESULT_DIR, exist_ok=True)

#
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]


graph = load_graph(embed=EMBEDDING_TYPE)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

results = []

print("Start responding to questions...")

# remove existing output file
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

for d in tqdm(data):
    q = d["text"]
    for chunk in graph.stream({"question": q}, config):
        for node, update in chunk.items():
            if node == "search_documents":
                id_scrores = update["retrieved_id_scores"]
            if node == "generate_answer":
                answer = update["answer"]
    
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"

    data = {
        "id": d["id"],
        "text": q,
        "answer": answer,
        "retrieved_docs": id_scrores,
    }

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        json.dump(data, f)
        f.write("\n")

    
