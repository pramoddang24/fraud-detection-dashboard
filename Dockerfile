FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade setuptools FIRST and install all dependencies
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip install --no-cache-dir "setuptools>=76.0.0" \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

EXPOSE 8000

# Use eventlet worker (as Flask-SocketIO officially recommends)
ENTRYPOINT ["sh", "-c"]
CMD ["exec gunicorn app:app -k eventlet -w 1 --bind 0.0.0.0:${PORT:-8000}"]
