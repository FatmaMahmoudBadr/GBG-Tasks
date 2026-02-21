import re
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

def clean_arabic(text):
    # Remove lines starting with ---
    text = re.sub(r'^---.*?---$', '', text, flags=re.MULTILINE)

    # Remove English letters
    text = re.sub(r'[A-Za-z]', '', text)

    # Remove file extensions like .txt
    text = re.sub(r'\.txt', '', text)

    # Remove diacritics
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)

    # Remove extra symbols except Arabic punctuation
    text = re.sub(r'[^ء-ي0-9\s\.\،\؟]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


print("Reading Arabic file...")
with open("arabic.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = clean_arabic(text)


print("Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", "؟"]
)

chunks = splitter.split_text(text)
print(f"Number of chunks: {len(chunks)}")
for chunk in chunks:
    print(chunk)

print("Loading BGE-M3 model...")
# model = SentenceTransformer("BAAI/bge-m3")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("Creating embeddings...")
embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    batch_size=16,
    show_progress_bar=True
)

embeddings = np.array(embeddings)


print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) 
index.add(embeddings)

print("RAG System Ready!\n")


def retrieve(query, top_k=4):
    instruction_query = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    query_embedding = model.encode(
        [instruction_query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = [chunks[i] for i in indices[0]]
    print("number of chunks: ", len(results))
    return results




genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0, 
    }
)


def generate_answer(query):
    contexts = retrieve(query)
    context_text = "\n\n".join(contexts)
    print("chunks: ",context_text)
    prompt = f"""
أنت مساعد ذكي تجيب فقط اعتماداً على السياق المعطى.
إذا لم تجد الإجابة في السياق، قل: "المعلومة غير متوفرة في النص".

السياق:
{context_text}

السؤال:
{query}

الإجابة:
"""

    response = llm.generate_content(prompt)
    return response.text


while True:
    question = input(" اسأل سؤالاً (اكتب exit للخروج): ")

    if question.lower() == "exit":
        break

    answer = generate_answer(question)

    print("\n=============================")
    print("الإجابة:\n")
    print(answer)
    print("=============================\n")
