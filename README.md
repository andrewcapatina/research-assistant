Steps to run the project:

1. Build the Docker container
    
    NOTE: you need to provide a hugging face token to access the model needed.

    DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=./hf_token.txt -t ai-agent:spark .  

2. Start the container and enter bash with GPU and ports enabled

    docker run -it --rm  --gpus all   -v ~/llama-models:/models \
    -v ~/agent-data:/app/data -v hf-cache:/root/.cache/huggingface \
    -p 11434:11434 -p 8501:8501   -w /app   ai-agent:spark bash



3. Run the setup script within the container to start the project.

    bash start_project.sh

Optionally, run convert_quant to convert ollama model to 4 bit quant.
