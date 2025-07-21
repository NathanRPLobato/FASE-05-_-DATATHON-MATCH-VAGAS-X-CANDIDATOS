# ml_api/Dockerfile

# 1) Use a slim Python base image
FROM python:3.11-slim

# 2) Instala libgomp (OpenMP) para suportar LightGBM, CatBoost, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 3) Define diretório de trabalho
WORKDIR /app

# 4) Copia o requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5) Copia todo o código da API
COPY ml_api ./ml_api

# 6) Expõe a porta padrão da Flask
EXPOSE 5000

# 7) Define o comando de inicialização
CMD ["python", "-m", "ml_api.main"]
