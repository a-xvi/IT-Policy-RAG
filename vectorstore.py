import os
import re
import sys
from functools import lru_cache

import chromadb

from file_reader import read_pdf_pages
from preprocessing import clean_text
from chunking import chunk_text
from embedding import embed_chunks


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE_DIR,
    "chroma_db"
)

COLLECTION_NAME = "it_policy"

# Policy PDF to index (override with the POLICY_PDF environment variable)
PDF_PATH = os.environ.get(
    "POLICY_PDF",
    os.path.join(
        BASE_DIR,
        "data",
        "dummy_it_policy(1).pdf"
    )
)


# Detect headings such as:
# 2.6.1. General Statements
# 4.2 Remote Access
SECTION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Za-z].+?)\s*$"
)

#opens the connect to chroma db
@lru_cache(maxsize=1)
def get_client():

    print("Loading ChromaDB...")

    return chromadb.PersistentClient(
        path=DB_PATH
    )

#grap the collection you want aka the policy after embeding it 
@lru_cache(maxsize=1)
def get_collection():

    client = get_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Vector database loaded. "
        f"Stored chunks: {collection.count()}"
    )

    return collection

#to check which sections this shunk is from 
def detect_sections(text, current_section="Unknown"):

    blocks = []

    current_lines = []
    section = current_section

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        match = SECTION_PATTERN.match(line)

        if match:

            # Save text belonging to previous section
            if current_lines:

                blocks.append(
                    {
                        "section": section,
                        "text": "\n".join(current_lines)
                    }
                )

                current_lines = []

            section_number = match.group(1)
            section_title = match.group(2)

            section = (
                f"{section_number} - "
                f"{section_title}"
            )

        current_lines.append(line)

    if current_lines:

        blocks.append(
            {
                "section": section,
                "text": "\n".join(current_lines)
            }
        )

    return blocks, section

#take the embeding and store it aka the main function
def rebuild_index(pdf_path):

    collection = get_collection()

    print("\nDeleting old documents...")

    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

    print("Reading PDF page by page...")

    pages = read_pdf_pages(
        pdf_path
    )

    print(
        "Pages:",
        len(pages)
    )

    all_chunks = []
    all_metadata = []

    current_section = "Unknown"

    for page_data in pages:

        page_number = page_data["page"]

        cleaned = clean_text(
            page_data["text"]
        )

        if not cleaned.strip():
            continue

        section_blocks, current_section = detect_sections(
            cleaned,
            current_section
        )

        for block in section_blocks:

            section = block["section"]

            chunks = chunk_text(
                block["text"],
                chunk_size=1200,
                overlap=200
            )

            for chunk in chunks:

                all_chunks.append(
                    chunk
                )

                all_metadata.append(
                    {
                        "page": page_number,
                        "section": section
                    }
                )

    print(
        "Number of chunks:",
        len(all_chunks)
    )

    print("\nCreating embeddings...")

    embeddings = embed_chunks(
        all_chunks
    )

    ids = [
        f"policy_chunk_{i}"
        for i in range(len(all_chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadata
    )

    print("\n==============================")
    print("INDEX COMPLETE")
    print("==============================")

    print(
        "Documents:",
        collection.count()
    )

    return collection

#embeds the users question
def search(query_embedding, top_k=4):

    collection = get_collection()

    return collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


def build_index_if_empty(pdf_path):
    """
    Build the index only when the collection has no chunks.

    Used at container startup so an existing index is reused.
    """

    collection = get_collection()

    if collection.count() > 0:

        print("Index already exists. Skipping indexing.")

        return collection

    if not os.path.exists(pdf_path):

        print(
            f"WARNING: Index is empty and PDF not found: {pdf_path}\n"
            "Place the policy PDF there (or set POLICY_PDF) "
            "and restart the container."
        )

        return collection

    return rebuild_index(
        pdf_path
    )


if __name__ == "__main__":

    if "--if-empty" in sys.argv:

        build_index_if_empty(
            PDF_PATH
        )

    else:

        rebuild_index(
            PDF_PATH
        )