FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

# Install dependencies and clean pip cache to save space
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces requires apps to listen on port 7860
EXPOSE 7860

# We use an extended 120 second timeout for the heavy machine learning predictions
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "2"]
