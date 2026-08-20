# Generation host notes

Shared generation for this project uses **Azure AI Foundry / Azure OpenAI**
or **local Ollama**. Google Colab + ngrok is no longer the documented path.

- Local Ollama or Azure on a laptop: see [`README.md`](../README.md) → **How to run**
- Azure-only generation (laptop or VM): see [`azurereadme.md`](../azurereadme.md)
- Optional GPU Ollama via Docker: [`docker-compose.ollama.yml`](../docker-compose.ollama.yml)

The notebook `notebooks/colab_gemma_ollama_host.ipynb` may still exist in the
repo for historical reference; do not treat it as the current setup.
