"""
Contract Seeder Script for LexiRAG.
Extracts the historical benchmark contract documents from the embedded Chroma database
and saves them to data/contracts/ for immediate ingestion and local development.
"""

import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "notebooks" / "contracts_vect_embedding" / "chroma.sqlite3"
OUT_DIR = BASE_DIR / "data" / "contracts"


def extract_and_seed():
    if not DB_PATH.exists():
        print(f"⚠️ Source database not found at {DB_PATH}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    query = """
    SELECT s.string_value as source, d.string_value as doc 
    FROM embedding_metadata s 
    JOIN embedding_metadata d ON s.id = d.id 
    WHERE s.key = 'source' AND d.key = 'chroma:document';
    """
    c.execute(query)
    rows = c.fetchall()

    by_source = {}
    for src, doc in rows:
        filename = Path(src).stem.replace(".docx", "") + ".txt"
        by_source.setdefault(filename, []).append(doc)

    print(f"📦 Extracting {len(rows)} clauses across {len(by_source)} legal contracts...")

    for filename, chunks in by_source.items():
        out_file = OUT_DIR / filename
        content = "\n\n--- [SECTION BREAK] ---\n\n".join(chunks)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ Saved '{filename}' ({len(chunks)} clauses, {len(content):,} characters)")

    conn.close()
    print(f"✨ Successfully seeded legal contracts into '{OUT_DIR}'!\n")


if __name__ == "__main__":
    extract_and_seed()
