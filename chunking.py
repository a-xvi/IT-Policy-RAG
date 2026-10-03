import re


def chunk_text(text, chunk_size=1200, overlap=150):
    """
    Split policy text into retrieval-friendly chunks.

    - Keeps chunks around 1200 characters.
    - Adds a small overlap.
    - Removes useless page-only chunks.
    - Avoids creating tiny chunks.
    """

    paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    # Remove paragraphs that are only page markers
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not re.fullmatch(
            r"=== PAGE \d+ ===",
            paragraph.strip()
        )
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # Skip useless tiny OCR fragments
        if len(paragraph) < 15:
            continue

        # ----------------------------------
        # Paragraph larger than chunk size
        # ----------------------------------
        if len(paragraph) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            start = 0

            while start < len(paragraph):

                end = min(
                    start + chunk_size,
                    len(paragraph)
                )

                piece = paragraph[start:end].strip()

                if len(piece) >= 50:
                    chunks.append(piece)

                if end == len(paragraph):
                    break

                start = end - overlap

            continue

        # ----------------------------------
        # Start current chunk
        # ----------------------------------
        if not current_chunk:

            current_chunk = paragraph

        # ----------------------------------
        # Add paragraph if it still fits
        # ----------------------------------
        elif (
            len(current_chunk)
            + len(paragraph)
            + 2
            <= chunk_size
        ):

            current_chunk += "\n\n" + paragraph

        # ----------------------------------
        # Save current chunk and start next
        # ----------------------------------
        else:

            chunks.append(current_chunk.strip())

            # Use complete words for overlap
            overlap_text = current_chunk[-overlap:]

            first_space = overlap_text.find(" ")

            if first_space != -1:
                overlap_text = overlap_text[first_space + 1:]

            current_chunk = (
                overlap_text.strip()
                + "\n\n"
                + paragraph
            )

    if current_chunk:
        current_chunk = current_chunk.strip()

        if len(current_chunk) >= 50:
            chunks.append(current_chunk)

    return chunks


if __name__ == "__main__":

    import os

    from file_reader import read_pdf
    from preprocessing import clean_text


    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    PDF_PATH = os.path.join(
        BASE_DIR,
        "data",
        "dummy_it_policy(1).pdf"
    )


    raw = read_pdf(PDF_PATH)

    cleaned = clean_text(raw)

    chunks = chunk_text(
        cleaned,
        chunk_size=1200,
        overlap=150
    )


    print("\n==============================")
    print("CHUNKING RESULT")
    print("==============================")

    print(
        f"Characters: {len(cleaned)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )


    for i, chunk in enumerate(chunks[:10]):

        print(
            f"\n--- CHUNK {i} ---"
        )

        print(
            f"Length: {len(chunk)}"
        )

        print(
            chunk[:500]
        )