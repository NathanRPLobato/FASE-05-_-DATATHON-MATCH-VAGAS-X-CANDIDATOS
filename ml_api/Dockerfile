FROM python:3.11-slim
WORKDIR /app

# Copia o requirements da raiz
COPY requirements.txt .

# Instala tudo de uma vez
RUN pip install --no-cache-dir -r requirements.txt

# Copia o backend
COPY ml_api ./ml_api

EXPOSE 5000
CMD ["python", "-m", "ml_api.main"]
