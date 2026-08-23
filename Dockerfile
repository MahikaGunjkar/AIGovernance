# Heinzy — reproducible image (Infra task S2)
# Pinned base + pinned deps so two machines produce identical dependency trees.
# This image runs the retrieval/ingest Python code. Generation is remote:
#   MODEL_PROVIDER=ollama      → MODEL_ENDPOINT (local/LAN Ollama)
#   MODEL_PROVIDER=azure_openai → AZURE_OPENAI_* (Azure AI Foundry / OpenAI)
# Weights are never bundled in this image.

FROM python:3.11-slim-bookworm

# System deps kept minimal. Add poppler/tesseract here later if PDF extraction
# (ingest M1) needs them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching. requirements.txt is fully pinned.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the source. store = Chroma thin client; webui = Flask for heinzy.webui.
COPY . .
RUN pip install --no-cache-dir -e ".[store,webui]"

# Pre-warm the fastembed model cache at build time (reads the tag straight out
# of the checked-in config.yaml, not hardcoded here) so containers don't pay a
# ~130MB download on first run -- fully self-contained image.
RUN python -c "from heinzy.common.config import load_config; from fastembed import TextEmbedding; c = load_config(); TextEmbedding(model_name=c.embed.model_tag)"

# Default command runs the retrieval smoke test so `docker run` proves the
# image works end to end. Override for real entrypoints.
CMD ["python", "scripts/smoke_retrieval.py"]
