# —— Stage 1: install dependencies ——
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only requirements first for better cache
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install numpy \
 && pip install scikit-learn==1.2.2 --only-binary=:all: \
 && pip install gevent --only-binary=:all: \
 && pip install -r requirements.txt --no-deps

# —— Stage 2: build final image ——
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code and models
COPY . .

EXPOSE 8000

# Launch the app with Gunicorn
CMD ["gunicorn", "backend.app:app", "--bind", "0.0.0.0:8000"]