FROM python:3.11-slim

WORKDIR /app

# For Installing dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For Copying project files
COPY . .

# For Creating data and model directories
RUN mkdir -p data model

# For Exposing ports
EXPOSE 8000 8501

# For Making start script executable
RUN chmod +x start.sh

# For Running both services
CMD ["bash", "start.sh"]
