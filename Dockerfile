# 1. Pick a Python base image that already has wheels for scikit-learn
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# 4. Copy your app code (and your models folder)
COPY . .

# 5. Expose your port (if needed)
EXPOSE 8000

# 6. Launch with Gunicorn
CMD ["gunicorn", "backend.app:app", "--bind", "0.0.0.0:8000"]
