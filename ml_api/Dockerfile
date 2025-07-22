#Use a slim Python base image
FROM python:3.11-slim

#Instala libgomp (OpenMP) para suportar LightGBM, CatBoost, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

#Define diretório de trabalho
WORKDIR /app

#Copia o requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#Copia todo o código da API
COPY ml_api ./ml_api

#Expõe a porta padrão da Flask
EXPOSE 5000

#Define o comando de inicialização
CMD ["python", "-m", "ml_api.main"]
