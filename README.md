Steps to run the project:

1. Build the Docker container
    
    docker build -t ai-agent:spark .

2. Start the container and enter bash with GPU and ports enabled

    docker run -it --rm --runtime nvidia --gpus all \
    -v ~/ollama-data:/root/.ollama -v ~/agent-data:/app/data \
    -p 11434:11434 -p 8501:8501 -w /app ai-agent:spark bash

3. Run the setup script within the container to start the project.

    bash start_project.sh
