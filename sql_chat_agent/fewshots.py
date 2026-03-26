import json
import faiss
from sentence_transformers import SentenceTransformer

with open("sql_chat_agent/data/fewshots.json") as f:
    examples = json.load(f)

questions = [ex["question"] for ex in examples]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


def retrieve_examples(query, k=2):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, k)

    return [examples[idx] for idx in indices[0]]


def few_shots(question):

    retrieved = retrieve_examples(question)

    formatted = ""
    for i, ex in enumerate(retrieved, start=1):
        formatted += f"EXAMPLE {i}\n"
        formatted += f"User Question: {ex['question']}\n"
        formatted += "SQL Query:\n"
        formatted += f"{ex['answer']}\n\n"

    return formatted