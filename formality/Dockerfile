FROM python:3.9-slim

WORKDIR /app

# Create cache directory and set permissions
RUN mkdir -p /tmp && \
    chmod -R 777 /tmp

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp
ENV HF_HOME=/tmp
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 7860

# Run Flask app directly
CMD ["python", "app.py"] 