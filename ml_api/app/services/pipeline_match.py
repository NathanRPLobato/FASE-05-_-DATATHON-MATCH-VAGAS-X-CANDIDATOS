# ml_api/app/services/pipeline_match.py
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from pathlib import Path
from ml_api.app.config import MODEL_DIR

# 1) Carrega os modelos (podem vir como dicts com a key 'model')
_oti_raw = joblib.load(Path(MODEL_DIR) / "match_otimista.pkl")
match_model_otimista = _oti_raw['model'] if isinstance(_oti_raw, dict) and 'model' in _oti_raw else _oti_raw

_pes_raw = joblib.load(Path(MODEL_DIR) / "match_pessimista.pkl")
match_model_pessimista = _pes_raw['model'] if isinstance(_pes_raw, dict) and 'model' in _pes_raw else _pes_raw

print("[INFO] Modelos de match carregados")


def ranquear_candidatos_para_vaga(vaga: dict, candidatos: list[dict]) -> list[dict]:
    """
    Gera ranking top10 de candidatos para a vaga, misturando:
      - score dos modelos otimista/pessimista
      - similaridade semântica SBERT
    Também elimina duplicatas por email.
    """

    # 1) Deduplica candidatos (mantém o primeiro de cada email)
    seen = set()
    unique_cands = []
    for c in candidatos:
        if c['email'] not in seen:
            seen.add(c['email'])
            unique_cands.append(c)

    # 2) Monta DataFrame com as features brutas dos modelos
    df = pd.DataFrame([
        {
            "cluster_cand_enc": cand["cluster"],
            "cluster_vaga_enc": vaga["cluster"],
            "cand_tec_sap":     int(cand.get("eh_sap", 0)),
            "vaga_tec_sap":     int(vaga.get("eh_sap", 0)),
        }
        for cand in unique_cands
    ])
    if df.empty:
        return []

    # 3) Reindexa colunas conforme feature_names_in_, se existir
    if hasattr(match_model_otimista, "feature_names_in_"):
        df = df.reindex(columns=match_model_otimista.feature_names_in_, fill_value=0)

    # 4) Predit probabilities
    p_oti = match_model_otimista.predict_proba(df)[:, 1]
    p_pes = match_model_pessimista.predict_proba(df)[:, 1]

    # 5) Coarse top20 (para limitar SBERT)
    scored = sorted(
        zip(unique_cands, p_oti, p_pes),
        key=lambda x: x[1] + x[2],
        reverse=True
    )[:20]

    # 6) Carrega SBERT e calcula similaridade vagas→candidatos
    from sentence_transformers import SentenceTransformer, util
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    emb_vaga = sbert.encode(vaga.get("descricao", ""), convert_to_tensor=True)

    final = []
    for cand, oti, pes in scored:
        texto = cand.get("texto_classificado", "")
        emb_cand = sbert.encode(texto, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_cand, emb_vaga))  # [0,1]

        # 70% weight do modelo + 30% de similaridade
        base = (oti + (1 - pes)) / 2
        compat = round((0.7 * base + 0.3 * sim) * 100, 2)

        final.append((cand, compat))

    # 7) Ordena novamente e pega top10
    final = sorted(final, key=lambda x: x[1], reverse=True)[:10]

    # 8) Formata saída
    ranking = []
    for cand, compat in final:
        ranking.append({
            "id":              cand["id"],
            "nome":            cand["nome"],
            "email":           cand["email"],
            "compatibilidade": compat
        })

    return ranking
