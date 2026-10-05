from file_reader import read_pdf
from preprocessing import clean_text
# quick check I used it for debuging to see if the chunk size is as wanted.

raw = read_pdf("data/dummy_it_policy.pdf")
cleaned = clean_text(raw)

paragraphs = cleaned.split("\n\n")

for i, para in enumerate(paragraphs):
    print(f"Paragraph {i}: {len(para)} characters")
    
longest = max(paragraphs, key=len)
print(f"\nlongest paragraph is {len(longest)} character long")