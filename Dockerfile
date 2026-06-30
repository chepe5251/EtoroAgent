FROM python:3.11-slim

WORKDIR /app

# System deps for httpx and PRAW
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY skills/ ./skills/
COPY main.py .

# Logs dir must exist inside container; bind-mounted from host
RUN mkdir -p logs

CMD ["python", "main.py"]
