import pandas as pd
from pathlib import Path

INPUT_FILE = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/outputs/clean_pages.csv")
OUTPUT_FILE = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/outputs/chunks.csv")

CHUNK_SIZE = 300  # words per chunk


def split_into_chunks(text, chunk_size=300):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def main():
    df = pd.read_csv(INPUT_FILE)

    all_chunks = []

    chunk_id = 1

    for _, row in df.iterrows():

        url = row["url"]
        title = row["title"]
        text = row["clean_text"]

        chunks = split_into_chunks(text, CHUNK_SIZE)

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "url": url,
                "title": title,
                "chunk_text": chunk
            })

            chunk_id += 1

    chunk_df = pd.DataFrame(all_chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Chunks saved to: {OUTPUT_FILE}")
    print(f"Total chunks created: {len(chunk_df)}")


if __name__ == "__main__":
    main()