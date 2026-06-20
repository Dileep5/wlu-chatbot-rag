import pandas as pd
from rank_bm25 import BM25Okapi

# Load chunks
df = pd.read_csv("outputs/chunks.csv")

documents = df["chunk_text"].fillna("").tolist()

# Tokenize
tokenized_docs = [doc.lower().split() for doc in documents]

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    print("\nTop BM25 Results:\n")

    for idx in top_indices:

        print("-" * 50)

        print("Title:")
        print(df.iloc[idx]["title"])

        print("\nSource:")
        print(df.iloc[idx]["url"])

        print("\nChunk:")
        print(df.iloc[idx]["chunk_text"][:600])

        print("\n")