# —— Stage 1: install dependencies ——
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

# ✅ FIX: Explicitly upgrade setuptools to a version that satisfies all sub-dependencies.
RUN pip install --upgrade pip wheel && \
    pip install --upgrade "setuptools>=75.8.2" && \
    pip install -r requirements.txt

# —— Stage 2: build final image ——
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8080

CMD sh -c "gunicorn app:app --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --bind 0.0.0.0:${PORT:-8080}"