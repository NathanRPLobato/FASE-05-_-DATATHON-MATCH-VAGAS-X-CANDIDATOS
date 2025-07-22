#Imagem base
FROM python:3.11-slim

#Define /app como diretório de trabalho
WORKDIR /app

#Copia requirements.txt da raiz
COPY requirements.txt .

#Instala dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#Copia todo o conteúdo de web_dash/ para /app no container
COPY web_dash/ .

#Expõe a porta 8000
EXPOSE 8000

#Roda o servidor Flask
CMD ["python", "app.py"]
