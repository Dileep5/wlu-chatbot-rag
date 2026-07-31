import pandas as pd
import chromadb

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load Chunk Data
# -----------------------------

df = pd.read_csv("outputs/chunks.csv")

documents = df["chunk_text"].fillna("").tolist()

# -----------------------------
# Build BM25 Index
# -----------------------------

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# Load Vector DB
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="data/vector_db")

collection = client.get_collection("wlu_chatbot_chunks")

# -----------------------------
# Hybrid Search
# -----------------------------

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    print("\nVECTOR RESULTS")
    print("=" * 60)

    query_embedding = model.encode(query).tolist()

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    vector_urls = set()

    for i in range(len(vector_results["documents"][0])):

        doc = vector_results["documents"][0][i]

        metadata = vector_results["metadatas"][0][i]

        url = metadata["url"]

        vector_urls.add(url)

        print(f"\nResult {i+1}")
        print("-" * 50)
        print(url)
        print()
        print(doc[:500])

    print("\n")
    print("BM25 RESULTS")
    print("=" * 60)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    bm25_urls = set()

    for rank, idx in enumerate(top_indices):

        url = df.iloc[idx]["url"]

        bm25_urls.add(url)

        print(f"\nResult {rank+1}")
        print("-" * 50)
        print(url)
        print()
        print(df.iloc[idx]["chunk_text"][:500])

    print("\n")
    print("HYBRID RESULTS")
    print("=" * 60)

    hybrid_urls = vector_urls.union(bm25_urls)

    for url in hybrid_urls:

        print(url)

    print(f"\nTotal unique results: {len(hybrid_urls)}")