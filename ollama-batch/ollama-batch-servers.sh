#!/bin/bash

# Constants
HOST="0.0.0.0"
BASE_PORT=11432
OLLAMA_BINARY="your ollama binary path"
LOG_DIR="ollama-server-logs"
SLEEP_INTERVAL=1

# Check if the number of GPUs is provided as an argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_gpus>"
    exit 1
fi

# Command-line argument
NUM_GPUS=$1

# Validate that NUM_GPUS is a positive integer
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
    echo "Error: <num_gpus> must be a positive integer."
    exit 1
fi

# Check if the Ollama binary exists
if [[ ! -x "$OLLAMA_BINARY" ]]; then
    echo "Error: Ollama binary not found or not executable at $OLLAMA_BINARY"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Start server instances
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_DIR}/${PORT}.log"

    # Environment variables
    export OLLAMA_LOAD_TIMEOUT="120m"
    export OLLAMA_KEEP_ALIVE="120m"
    export OLLAMA_NUM_PARALLEL="16"
    export OLLAMA_HOST="${HOST}:${PORT}"
    export CUDA_VISIBLE_DEVICES="$i"

    # Start server with nohup and log output
    nohup "$OLLAMA_BINARY" serve > "$LOG_FILE" 2>&1 &

    if [[ $? -eq 0 ]]; then
        echo "Started server instance $i on port ${PORT}, logging to ${LOG_FILE}"
    else
        echo "Error: Failed to start server instance $i on port ${PORT}"
    fi

    # Sleep interval between starting instances
    sleep "$SLEEP_INTERVAL"
done

echo "All server instances started successfully."
