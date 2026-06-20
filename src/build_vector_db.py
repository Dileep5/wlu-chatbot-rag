import pandas as pd
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/outputs/chunks.csv")
DB_DIR = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/data/vector_db")

# A small local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    # Load chunk data
    df = pd.read_csv(INPUT_FILE)

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Create ChromaDB persistent client
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Create or get collection
    collection = client.get_or_create_collection(name="wlu_chatbot_chunks")

    # Prepare data
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for _, row in df.iterrows():
        chunk_id = str(row["chunk_id"])
        chunk_text = str(row["chunk_text"])
        title = str(row["title"])
        url = str(row["url"])

        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append({
            "title": title,
            "url": url
        })

        emb = model.encode(chunk_text).tolist()
        embeddings.append(emb)

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Saved {len(ids)} chunks to ChromaDB.")
    print(f"Vector DB location: {DB_DIR}")


if __name__ == "__main__":
    main()