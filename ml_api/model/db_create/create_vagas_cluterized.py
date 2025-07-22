import os
import json
import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk import download

# Baixar stopwords se ainda não tiver
download('stopwords')

# Caminhos
CAMINHO_JSON = "ml_api/data/extracted/vagas.json"
CAMINHO_MODELO = "ml_api/model/vaga_cluster_model.joblib"
CAMINHO_SAIDA = "ml_api/data/refined/vagas.csv"

# Carrega o modelo de clustering treinado
modelo = joblib.load(CAMINHO_MODELO)
print("Modelo carregado com sucesso")

# Carrega o JSON com as vagas
with open(CAMINHO_JSON, encoding="utf-8") as f:
    vagas_raw = json.load(f)

linhas_csv = []

for vaga_id, vaga in vagas_raw.items():
    info = vaga.get("informacoes_basicas", {})
    perfil = vaga.get("perfil_vaga", {})

    titulo = info.get("titulo_vaga", "") or ""
    cliente = info.get("cliente", "") or ""
    vaga_sap = info.get("vaga_sap", "")
    nivel = perfil.get("nivel profissional", "") or ""
    descricao = perfil.get("descricao_atividades", "") or ""
    competencias = perfil.get("competencia_tecnicas_e_comportamentais", "") or ""

    texto_completo = f"{titulo} {descricao} {competencias}".strip()

    # Ignora vagas com descrição fraca ou vazia
    if not texto_completo or len(texto_completo.split()) < 5:
        continue

    # Tenta prever cluster
    try:
        cluster = modelo.predict([texto_completo])[0]
    except Exception as e:
        print(f"Erro ao classificar vaga {vaga_id}: {e}")
        continue

    # Verifica se a classificação retornou válida
    if cluster is None or str(cluster).strip() == "":
        continue

    linha = {
        "vaga_id": vaga_id,
        "cliente": cliente,
        "titulo": titulo,
        "nível": nivel,
        "é_sap": 1 if vaga_sap.lower() == "sim" else 0,
        "descricao": descricao,
        "competencias": competencias,
        "classification": cluster
    }
    linhas_csv.append(linha)

# Salva no CSV
df = pd.DataFrame(linhas_csv)
os.makedirs(os.path.dirname(CAMINHO_SAIDA), exist_ok=True)
df.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8-sig")
print(f"CSV com {len(df)} vagas salvas em: {CAMINHO_SAIDA}")