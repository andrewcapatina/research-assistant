Steps to run the project:

1. Build the Docker container
    
    docker build -t ai-agent:spark .

2. Start the container and enter bash with GPU and ports enabled

    docker run -it --rm --runtime nvidia --gpus all \
    -v ~/ollama-data:/root/.ollama -v ~/agent-data:/app/data \
    -p 11434:11434 -p 8501:8501 -w /app ai-agent:spark bash

3. Start ollama and pull a model

    ollama serve &
    ollama pull llama3.1:8b

4. Run the agent to collect the papers, and summarize.

    python3 agent.py

5. Start the web UI to display the summaries.

    streamlit run gui.py
