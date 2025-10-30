# Base from the Ollama image for JetPack 7 on Thor (ARM64, CUDA 13.0)
FROM ubuntu:24.04

# Set working directory
WORKDIR /app

# NVPL install: https://developer.nvidia.com/nvpl-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
# For sentence_transformers lib
RUN apt-get update && apt-get install -y wget gnupg ca-certificates && \
    wget https://developer.download.nvidia.com/compute/nvpl/25.5/local_installers/nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb && \
    dpkg -i nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb && \
    cp /var/nvpl-local-repo-ubuntu2404-25.5/nvpl-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install nvpl

# CUDSS install:
# For sentence_transformers lib
RUN wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb && \
    dpkg -i cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb && \
    cp /var/cudss-local-repo-ubuntu2404-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cudss

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-13.0/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64

RUN apt-get -y install curl
# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies (same as before)
# Install PyTorch with CUDA 13.0 support
RUN apt-get update && apt-get install -y python3-pip && \
    python3 -m pip install --no-cache-dir --break-system-packages \
    pypdf2 \
    requests \
    streamlit \
    pandas \
    faiss-cpu \
    sentence-transformers \
    langgraph \
    langchain \
    langchain-ollama \
    langchain-community \
    arxiv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages torch --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/

# Copy your code files into the container (add them to your project dir first)
COPY start_project.sh shared_llm.py web_scraping.py summarize.py storage.py agent.py \
     retrieve.py manual_papers.json gui.py memory.py planner_agent.py ./

# Expose Ollama API port if needed (default 11434)
EXPOSE 11434

# Default command: Run the Ollama server in background, then keep container alive
CMD ["bash", "-c", "ollama serve & sleep infinity"]
