from rag import load_graph
import uuid
import json
from pathlib import Path
from tqdm import tqdm
import os
import argparse
# ===================
#      Config
# ===================
DATA_FILE = "data/validation.jsonl"
RESULT_DIR = 'results'
EMBEDDING_TYPE = "dense"  # static, dense, minilm

def run_in_batch(data_file: str = DATA_FILE ,embed_type: str = EMBEDDING_TYPE):
    """
    Run batch generation for questions in DATA_FILE and save results to OUTPUT_FILE.
    """

    data_file = Path(data_file)
    output_file = Path(f"{RESULT_DIR}/{data_file.stem}_{embed_type}.jsonl")

    # create result directory if not exists
    os.makedirs(RESULT_DIR, exist_ok=True)

    #
    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]


    graph = load_graph(embed=embed_type)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Start responding to questions...")

    # remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    for d in tqdm(data):
        q = d["text"]
        for chunk in graph.stream({"question": q}, config):
            for node, update in chunk.items():
                if node == "search_documents":
                    id_scrores = update["retrieved_id_scores"]
                if node == "generate_answer":
                    answer = update["answer"]
        
        mode = "a" if os.path.exists(output_file) else "w"

        data = {
            "id": d["id"],
            "text": q,
            "answer": answer,
            "retrieved_docs": id_scrores,
        }

        with open(output_file, mode, encoding="utf-8") as f:
            json.dump(data, f)
            f.write("\n")

    print('Answers are saved to {}'.format(output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch generate answers for questions.')
    parser.add_argument('-f','--file', type=str, default=DATA_FILE, help='Path to the input data file.')
    parser.add_argument('-e','--embed', type=str, default=EMBEDDING_TYPE, help='Embedding type: static, dense, minilm.')

    args = parser.parse_args()
    data_file = args.file
    embed = args.embed
    run_in_batch(data_file=data_file, embed_type=embed)


