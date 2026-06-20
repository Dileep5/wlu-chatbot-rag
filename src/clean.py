import pandas as pd
import re
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "raw_pages.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "clean_pages.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def main():
    df = pd.read_csv(INPUT_FILE)

    # Create cleaned text column
    df["clean_text"] = df["text"].apply(clean_text)

    # Save cleaned output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()