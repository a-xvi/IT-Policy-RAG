#!/bin/sh
# Container startup:
# 1. Download the Ollama model if it is missing
# 2. Build the vector index if chroma_db is empty
# 3. Start the Streamlit app
# Steps 1 and 2 log failures but do not block the UI.

python pull_model.py
python vectorstore.py --if-empty

exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
