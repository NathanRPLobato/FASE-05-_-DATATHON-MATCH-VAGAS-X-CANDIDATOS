FROM python:3.11-slim
WORKDIR /app

# Usa o mesmo requirements.txt da raiz
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia o frontend
COPY web_dash .

EXPOSE 8000
CMD ["python", "app.py"]
