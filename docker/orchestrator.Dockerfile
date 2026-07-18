# Orchestrator: LangGraph spine + RAG embeddings (BGE-M3 rides in this
# container's 6 GB line alongside FastAPI). RUN ON SPARK.
# Not in the original 5-file list — added because the orchestrator needs its
# own image; documented in the README repo map.
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN pip install --no-cache-dir \
      fastapi uvicorn[standard] python-multipart httpx \
      langgraph openai qdrant-client FlagEmbedding

# fail the BUILD (not first serve) if pip replaced the NGC torch with a CPU wheel
# (FlagEmbedding's dep tree is the likeliest offender here)
RUN python -c "import torch; assert torch.version.cuda, 'pip clobbered the NGC torch with a CPU build'"

WORKDIR /app
COPY ci/ ci/
COPY serving/ serving/
COPY training/configs/ training/configs/
ENV PYTHONPATH=/app

CMD python ci/preflight.py --check torch && \
    uvicorn serving.orchestrator.app:app --host 0.0.0.0 --port 8080
