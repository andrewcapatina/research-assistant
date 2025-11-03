# Base ubuntu image the DGX spark uses
FROM nvcr.io/nvidia/vllm:25.09-py3

# Set working directory
WORKDIR /app

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pypdf2 \
    requests \
    streamlit \
    pandas \
    faiss-cpu \
    sentence-transformers \
    langgraph \
    huggingface_hub \
    arxiv \
    transformers \
    accelerate \ 
    autoawq \
    bitsandbytes && \
    rm -rf /var/lib/apt/lists/*

# Log in to HF CLI and download the model, and cache it.
RUN --mount=type=secret,id=hf_token \
    huggingface-cli login --token $(cat /run/secrets/hf_token) --add-to-git-credential && \
    huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --cache-dir /root/.cache/huggingface/hub \
        --local-dir /app/models/llama-3.1-8b-instruct \
        --local-dir-use-symlinks False \
        --cache-dir /root/.cache/huggingface

COPY start_project.sh shared_llm.py web_scraping.py summarize.py \
    storage.py retrieve.py manual_papers.json gui.py memory.py planner_agent.py \
     benchmarking.py convert_quant.py ./

# Expose Ollama API port if needed (default 11434)
EXPOSE 11434
