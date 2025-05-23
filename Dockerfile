FROM python:3.13
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN apt-get update &&\
    apt-get install -y swig &&\
    pip install --no-cache-dir -r requirements.txt
COPY . /app/
EXPOSE 8000
CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]