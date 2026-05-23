FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data and model directories
RUN mkdir -p data model

# Expose ports
EXPOSE 8000 8501

# Make start script executable
RUN chmod +x start.sh

# Run both services
CMD ["bash", "start.sh"]
