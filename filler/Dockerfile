FROM python:3.10

WORKDIR /app

# Install system dependencies including ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask openai-whisper torch torchaudio

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]