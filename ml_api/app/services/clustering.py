# ml_api/app/services/clustering.py
# -*- coding: utf-8 -*-

import joblib
from pathlib import Path
from ml_api.app.config import Config

# --- Carrega modelo e vetor de candidatos ---
try:
    _cand_raw = joblib.load(Path(Config.CAND_CLUSTER_PATH))
    cand_vectorizer = _cand_raw['vectorizer']
    cand_model      = _cand_raw['model']
    print('[INFO] Modelo de clustering de candidatos carregado')
except Exception as e:
    print(f"[ERRO] Falha ao carregar clustering candidatos: {e}")
    raise

# --- Carrega modelo e vetor de vagas ---
try:
    _vaga_raw = joblib.load(Path(Config.VAGA_CLUSTER_PATH))
    vaga_vectorizer = _vaga_raw['vectorizer']
    vaga_model      = _vaga_raw['model']
    print('[INFO] Modelo de clustering de vagas carregado')
except Exception as e:
    print(f"[ERRO] Falha ao carregar clustering vagas: {e}")
    raise


def cluster_candidato(features: dict) -> int:
    """
    Extrai o texto_classificado, transforma em embedding e prediz o cluster.
    """
    texto = features.get('texto_classificado', '')
    emb = cand_vectorizer.transform([texto])
    cluster = int(cand_model.predict(emb)[0])
    print(f"[DEBUG] cluster_candidato -> '{texto}' -> {cluster}")
    return cluster


def cluster_vaga(features: dict) -> int:
    """
    Junta titulo+descricao, transforma em embedding e prediz o cluster.
    """
    titulo    = features.get('titulo', '')
    descricao = features.get('descricao', '')
    texto = f"{titulo} {descricao}".strip()
    emb = vaga_vectorizer.transform([texto])
    cluster = int(vaga_model.predict(emb)[0])
    print(f"[DEBUG] cluster_vaga -> '{texto}' -> {cluster}")
    return cluster
