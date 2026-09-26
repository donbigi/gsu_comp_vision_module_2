# Camera calibration + perspective-projection measurement demo.
# Serves the Flask web app on port 8000.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System libraries needed by OpenCV (headless) on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (calibration/ and measurement/ images included).
COPY . .

# Run as an unprivileged user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "app.py"]
