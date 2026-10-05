FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

COPY . .

ENV HF_HUB_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8501

# Build the vector index on first start (skipped if chroma_db already has data),
# then launch the app. Indexing failures are logged but do not block the UI.
CMD ["sh", "-c", "python vectorstore.py --if-empty; exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]