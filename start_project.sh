#!/bin/bash

# Script to start Ollama, download model, run agent.py, and launch gui.py
# Assumes you're running in the Docker container or host with Ollama/Python installed

# Configurable variables
MODEL="llama3.1:8b"  # Change to your preferred model, e.g., "llama3.1:8b-q4_0" for quantized
DB_PATH="/app/data/papers.db"  # Ensure matches your code

# Start Ollama server in background if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5  # Wait for server to initialize
else
    echo "Ollama server already running."
fi

# Download/pull the model if not present
if ! ollama list | grep -q "$MODEL"; then
    echo "Downloading model $MODEL..."
    ollama pull "$MODEL"
else
    echo "Model $MODEL already downloaded."
fi

# Run agent.py to fetch and summarize (add --no-fetch if you don't want auto-pull)
echo "Running agent.py..."
python3 agent.py  # Or add args like --no-fetch

# Launch Streamlit GUI
echo "Starting GUI..."
streamlit run gui.py