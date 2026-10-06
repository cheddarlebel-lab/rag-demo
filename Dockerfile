# Runs the pipeline in a container; ollama stays on the host (it needs the GPU / Metal).
#   docker build -t rag-demo .
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 -v "$PWD/data:/app/data" rag-demo crawl
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 -v "$PWD/data:/app/data" rag-demo index
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 -v "$PWD/data:/app/data" rag-demo ask "How do I limit upload size?"
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://host.docker.internal:11434
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag_demo ./rag_demo
COPY eval ./eval
RUN useradd --create-home app && mkdir -p data && chown -R app /app
USER app
ENTRYPOINT ["python", "-m", "rag_demo"]
CMD ["--help"]
