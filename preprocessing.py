import re


def remove_toc_lines(text):
    """
    Remove TOC filler lines consisting mostly of dots/numbers.
    """

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped and re.fullmatch(r"[.\d\s]+", stripped):
            if "." in stripped:
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


def remove_noise_lines(text):
    """
    Remove obvious OCR noise while preserving real policy text.
    """

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        # Standalone page numbers
        if re.fullmatch(r"\d{1,3}", stripped):
            continue

        # Standalone bullets / OCR artifacts
        if re.fullmatch(r"[•▪‐\-–—✓✖:.\s]+", stripped):
            continue

        # OCR sometimes turns bullets into "o"
        if stripped.lower() == "o":
            continue

        # Repeated OCR bullet characters
        if re.fullmatch(r"[o•▪]+", stripped.lower()):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove TOC
    text = remove_toc_lines(text)

    # Remove obvious OCR noise
    text = remove_noise_lines(text)

    # Remove trailing spaces
    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    # Too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


if __name__ == "__main__":

    from file_reader import read_pdf

    raw = read_pdf("data/dummy_it_policy(1).pdf")

    print("\nRAW CHARACTERS:", len(raw))

    cleaned = clean_text(raw)

    print("CLEANED CHARACTERS:", len(cleaned))

    print("\n==============================")
    print("FIRST 5000 CHARACTERS")
    print("==============================\n")

    print(cleaned[:5000])