FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip install --no-cache-dir "setuptools<81.0.0" \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# The critical line—use CMD in shell form or as an array with sh -c:
CMD ["sh", "-c", "gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 --bind 0.0.0.0:${PORT:-8080} app:app"]
