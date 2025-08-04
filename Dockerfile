FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Force upgraded pip + setuptools first
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip install --no-cache-dir "setuptools>=76.0.0"

# Now install the rest
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT ["sh", "-c"]
CMD ["exec gunicorn app:app --worker-class eventlet --bind 0.0.0.0:${PORT:-8000}"]
