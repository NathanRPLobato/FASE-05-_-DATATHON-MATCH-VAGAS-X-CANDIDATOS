# web_dash/Dockerfile

# 1) Imagem base
FROM python:3.11-slim

# 2) Define /app como diretório de trabalho
WORKDIR /app

# 3) Copia requirements.txt da raiz
COPY requirements.txt .

# 4) Instala dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5) Copia todo o conteúdo de web_dash/ para /app no container
COPY web_dash/ .

# 6) Expõe a porta 8000
EXPOSE 8000

# 7) Roda o servidor Flask
CMD ["python", "app.py"]
