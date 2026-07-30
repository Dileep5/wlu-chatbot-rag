import pandas as pd
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "chunks.csv"
DB_DIR = BASE_DIR / "data" / "vector_db"

# A small local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "wlu_chatbot_chunks"


def main():
    # Load chunk data
    df = pd.read_csv(INPUT_FILE)

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Create ChromaDB persistent client
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuilding should not accumulate stale/duplicate entries from a
    # previous run - matches build_faculty_vector_db.py's existing
    # pattern. Without this, ChromaDB's add() silently no-ops on ids
    # that already exist (confirmed empirically) rather than erroring or
    # overwriting, so a weekly refresh would never actually update any
    # previously-seen chunk's content - only chunk_id positions beyond
    # the prior run's max would ever get written.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create or get collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

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