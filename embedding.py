import os
from functools import lru_cache

#to stay offline to keep all models local to keep everything confi
os.environ["HF_HUB_OFFLINE"] = "1"
#so the model doesnt work paralell as my cpu is not that strong
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "all-MiniLM-L6-v2"
)

#last using used to keep the anwsers for more recent used or asked question 
@lru_cache(maxsize=1)
def load_embedding_model():

    print("Loading embedding model...")

    # Docker / local project model folder
    if os.path.exists(LOCAL_MODEL_PATH):

        print(f"Using local model: {LOCAL_MODEL_PATH}")

        model = SentenceTransformer(
            LOCAL_MODEL_PATH,
            local_files_only=True
        )

    # Windows Hugging Face cache
    else:

        print("Using cached Hugging Face model...")

        model = SentenceTransformer(
            MODEL_NAME,
            local_files_only=True
        )

    print("Embedding model loaded.")

    return model

# embed chunk trnasfprm it to numbers
def embed_chunks(chunks):

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32
    )

    return embeddings


#tuens user question to numbers aka vectors
def embed_query(query):

    model = load_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embedding


if __name__ == "__main__":

    test_text = [
        "Password requirements",
        "Remote access policy"
    ]

    vectors = embed_chunks(test_text)

    print("\nEmbedding test successful.")
    print("Vectors:", len(vectors))
    print("Vector dimension:", len(vectors[0]))