# ml_api/model/train_unsupervised_candidates.py

import os
import json
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import download

# Baixa stopwords do NLTK se necessário
download('stopwords')
stop_words = set(stopwords.words('portuguese'))
stemmer = SnowballStemmer('portuguese')

JSON_PATH = os.path.join("ml_api", "data", "extracted", "applicants.json")
MODEL_PATH = os.path.join("ml_api", "model", "candidato_cluster_model.joblib")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zà-ú0-9\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def load_and_prepare_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    all_texts = []
    for entry in data.values():
        cv_text = entry.get("cv_pt", "")
        prof = entry.get("informacoes_profissionais", {})
        form = entry.get("formacao_e_idiomas", {})

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
        all_texts.append(clean_text(combined))
    return all_texts

def train_and_save_model(texts, path, n_clusters=8):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(texts)

    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
    model.fit(X)

    joblib.dump({"model": model, "vectorizer": vectorizer}, path)
    print(f"Modelo salvo em {path}")

def main():
    print("Carregando e limpando dados...")
    texts = load_and_prepare_data(JSON_PATH)
    print("Treinando modelo não supervisionado (candidatos)...")
    train_and_save_model(texts, MODEL_PATH, n_clusters=8)
    print("Treinamento concluído e modelo salvo.")

if __name__ == "__main__":
    main()
