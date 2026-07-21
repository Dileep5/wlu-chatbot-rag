import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/data/faculty.db")
VECTOR_DB_DIR = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/data/vector_db")

# Same local embedding model already used for the general-page collection.
MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "wlu_faculty_research"


def main():

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
    SELECT source_url, name, research_interests
    FROM faculty
    WHERE research_interests IS NOT NULL
      AND TRIM(research_interests) != ''
    """)

    rows = cursor.fetchall()

    conn.close()

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Rebuilding should not accumulate stale/duplicate entries from a
    # previous run - start the collection fresh each time this is run.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for source_url, name, research_interests in rows:

        # source_url as the Chroma document id - stable across rebuilds,
        # unlike faculty.id (an AUTOINCREMENT key that load_faculty.py
        # reassigns on every full reload, since it does DELETE + re-insert
        # rather than updating rows in place). A stale id here would
        # silently make every research-topic query return no results,
        # which is exactly what happened before this fix.
        ids.append(source_url)
        documents.append(research_interests)

        # Minimal metadata: just enough to re-fetch the authoritative row
        # from faculty.db. SQLite remains the sole source of truth for
        # every displayed fact (title, department, contact info, etc.) -
        # nothing here duplicates that. source_url is the only persistent
        # key; name is kept purely as a convenience label, not an
        # identifier - re-fetching always goes by source_url.
        metadatas.append({
            "source_url": source_url,
            "name": name,
        })

        embeddings.append(model.encode(research_interests).tolist())

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Saved {len(ids)} faculty research records to ChromaDB.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector DB location: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
