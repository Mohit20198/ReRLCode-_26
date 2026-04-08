FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────
COPY . .

# ── __init__ files for imports ────────────────────────────────
RUN find /app -type d -exec touch {}/__init__.py \; 2>/dev/null || true

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "dashboard.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
