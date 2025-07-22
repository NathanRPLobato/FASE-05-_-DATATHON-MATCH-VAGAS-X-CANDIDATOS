import os
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk import download

download('stopwords')

CAMINHO_JSON = "ml_api/data/extracted/vagas.json"
CAMINHO_SAIDA_MODELO = "ml_api/model/vaga_cluster_model.joblib"

def carregar_vagas(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)

def preprocessar_vagas(vagas_raw):
    vagas_limpa = []
    for vaga_id, vaga in vagas_raw.items():
        info = vaga.get("informacoes_basicas", {})
        perfil = vaga.get("perfil_vaga", {})

        titulo = info.get("titulo_vaga", "") or ""
        descricao = perfil.get("descricao_atividades", "") or ""
        competencias = perfil.get("competencia_tecnicas_e_comportamentais", "") or ""

        texto_completo = f"{titulo} {descricao} {competencias}".strip()

        # Filtros: vagas muito curtas ou completamente vazias
        if not texto_completo or len(texto_completo.split()) < 5:
            continue

        vagas_limpa.append(texto_completo)

    # Remove duplicadas e vazias
    vagas_limpa = [t for t in pd.Series(vagas_limpa).drop_duplicates().dropna() if t.strip()]
    return vagas_limpa

def train_and_save_model(texts, path, n_clusters=6):
    stopwords_pt = stopwords.words("portuguese")
    vectorizer = TfidfVectorizer(stop_words=stopwords_pt, max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(texts)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    model.fit(X)

    joblib.dump({"model": model, "vectorizer": vectorizer}, path)
    print(f"Modelo salvo em {path}")

def main():
    print("Lendo vagas...")
    vagas_raw = carregar_vagas(CAMINHO_JSON)
    textos = preprocessar_vagas(vagas_raw)

    if not textos:
        print("Nenhuma vaga válida encontrada após limpeza.")
        exit()

    print("Treinando modelo não supervisionado (vagas)...")
    train_and_save_model(textos, CAMINHO_SAIDA_MODELO, n_clusters=6)
    print("Treinamento concluído e modelo salvo.")

if __name__ == "__main__":
    main()
