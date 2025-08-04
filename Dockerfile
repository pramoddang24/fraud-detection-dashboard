# —— Stage 1: install dependencies ——
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy requirements and install all packages (wheels only)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# —— Stage 2: build final image ——
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and executables from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code and models
COPY . .

EXPOSE 8000

# Launch the app with Gunicorn
CMD ["gunicorn", "backend.app:app", "--worker-class", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "--bind", "0.0.0.0:8000"]