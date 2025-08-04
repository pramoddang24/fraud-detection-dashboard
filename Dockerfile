# —— Stage 1: install dependencies ——
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# —— Stage 2: build final image ——
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ... (rest of the file is the same) ...

COPY . .

EXPOSE 8080

# ✅ FIX: Using sh -c to enable port fallback/default behavior correctly
CMD sh -c "gunicorn app:app --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --bind 0.0.0.0:${PORT:-8080}"
