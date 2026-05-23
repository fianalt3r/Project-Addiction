#!/bin/bash

echo "🚀 Starting Social Media Addiction Analysis App..."

# Start FastAPI backend
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload &

# Wait for backend to be ready
sleep 3

# Start Streamlit frontend
echo "🖥️  Starting Streamlit frontend on port 8501..."
streamlit run src/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true

wait
