import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/vector_db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection("wlu_chatbot_chunks")

# Ask a question
query = input("Ask a question: ")

# Convert question into embedding
query_embedding = model.encode(query).tolist()

# Search similar chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nTop Matching Chunks:\n")

for i in range(len(results["documents"][0])):

    print(f"Result {i+1}")
    print("-" * 50)

    print("Chunk:")
    print(results["documents"][0][i])

    print("\nSource:")
    print(results["metadatas"][0][i]["url"])

    print("\n")