# Base from the Ollama image for JetPack 7 on Thor (ARM64, CUDA 13.0)
FROM ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04

# Set working directory
WORKDIR /app

# Install Python dependencies (same as before)
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir \
    pypdf2 \
    requests \
    streamlit \
    pandas \
    langchain \
    langchain-ollama \
    langchain-community \
    arxiv && \
    rm -rf /var/lib/apt/lists/*

# Copy your code files into the container (add them to your project dir first)
COPY start_project.sh web_scraping.py summarize.py storage.py agent.py retrieve.py manual_papers.json gui.py ./

# Expose Ollama API port if needed (default 11434)
EXPOSE 11434

# Default command: Run the Ollama server in background, then keep container alive
CMD ["bash", "-c", "ollama serve & sleep infinity"]
