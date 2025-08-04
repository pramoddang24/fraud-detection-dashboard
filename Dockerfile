FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip wheel \
 && pip install "setuptools>=76.0.0" \
 && pip install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000

ENTRYPOINT ["sh", "-c"]
CMD ["exec gunicorn app:app --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --bind 0.0.0.0:${PORT:-8000}"]
