FROM python:3.12.3-slim

WORKDIR /app

COPY main.py /app
COPY prompt.txt /app
COPY requirements.txt /app
COPY .env /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"] 