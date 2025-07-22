import os
import json
import re
import csv
import joblib
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import download
from unidecode import unidecode

# Configurações
download('stopwords')
stop_words = set(stopwords.words('portuguese'))
stemmer = SnowballStemmer('portuguese')

# Caminhos
JSON_PATH = os.path.join("ml_api", "data", "extracted", "applicants.json")
MODEL_PATH = os.path.join("ml_api", "model", "unsupervised_candidate_model.joblib")
CSV_PATH = os.path.join("ml_api", "data", "refined", "candidatos.csv")

def clean_text(text):
    if not text:
        return ""
    text = unidecode(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def load_candidates(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def preprocess_candidate(candidate):
    try:
        cv_text = candidate.get("cv_pt", "")
        prof = candidate.get("informacoes_profissionais", {})
        form = candidate.get("formacao_e_idiomas", {})

        combined = " ".join([
            cv_text,
            prof.get("titulo_profissional", ""),
            prof.get("area_atuacao", ""),
            prof.get("conhecimentos_tecnicos", ""),
            form.get("nivel_academico", ""),
            form.get("nivel_ingles", ""),
            form.get("nivel_espanhol", ""),
            form.get("outro_idioma", "")
        ])
        cleaned = clean_text(combined)
        return cleaned if cleaned.strip() else None
    except Exception:
        return None

def classificar_candidatos(dados, modelo):
    resultados = []
    total = len(dados)

    for idx, (id_cand, candidato) in enumerate(dados.items(), 1):
        try:
            texto = preprocess_candidate(candidato)
            if not texto:
                continue

            vector = modelo["vectorizer"].transform([texto])
            cluster_id = modelo["model"].predict(vector)[0]

            resultados.append({
                "id": id_cand,
                "nome": candidato.get("infos_basicas", {}).get("nome", ""),
                "email": candidato.get("infos_basicas", {}).get("email", ""),
                "cluster": cluster_id,
                "texto_classificado": texto 
            })
        except Exception as e:
            print(f"Erro ao classificar candidato {id_cand}: {e}")
    
    return resultados

def salvar_csv(candidatos, path):
    df = pd.DataFrame(candidatos)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Arquivo salvo em {path}")

def main():
    print("Carregando dados e modelo...")
    candidatos_json = load_candidates(JSON_PATH)
    modelo = joblib.load(MODEL_PATH)

    print("Classificando candidatos...")
    candidatos_classificados = classificar_candidatos(candidatos_json, modelo)

    print(f"Candidatos classificados: {len(candidatos_classificados)}")
    salvar_csv(candidatos_classificados, CSV_PATH)

if __name__ == "__main__":
    main()

