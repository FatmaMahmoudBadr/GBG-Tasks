from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import json
import faiss

with open("langchain_sql_agent/fewshots.json") as f:
    examples = json.load(f)

questions  = [ex["question"] for ex in examples]

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

    results = []
    for idx in indices[0]:
        results.append(examples[idx])

    return results

def few_shots(question):
    formatted = ""
    retrieved_examples = retrieve_examples(question)
    for i, ex in enumerate(retrieved_examples, start=1):
        formatted += f"EXAMPLE {i}\n"
        formatted += f"User Question: {ex['question']}\n"
        formatted += "SQL Query:\n"
        formatted += f"{ex['answer']}\n"

    return formatted


# few_shots = few_shots("How many customers in United States?")

# print (few_shots)