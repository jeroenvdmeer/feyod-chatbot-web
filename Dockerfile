FROM python:3-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt .

RUN apt-get update &&\
    apt-get install -y --no-install-recommends swig &&\
    apt-get upgrade -y &&\
    rm -rf /var/lib/apt/lists/* &&\
    pip install --no-cache-dir -r requirements.txt

COPY . /app/
EXPOSE 8000

# Use a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]