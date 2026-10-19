FROM python:3.10-slim

# espeak-ng provides the phonemization kokoro needs; libsndfile1 is the WAV
# backend soundfile loads at import time. Both are required at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch first, off the official CPU index, so the image never pulls
# the ~2GB CUDA wheels; kokoro + fastapi land on top of it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# bake the stock Kokoro-82M weights, config, and the two default female
# voicepacks (en af_heart, hi hf_alpha) so cold start skips the HuggingFace
# download (otherwise the first request can time out on Render)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
files = ['kokoro-v1_0.pth', 'config.json', 'voices/af_heart.pt', 'voices/hf_alpha.pt']; \
[hf_hub_download('hexgrad/Kokoro-82M', f) for f in files]"

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
