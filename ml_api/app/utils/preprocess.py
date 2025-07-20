# ml_api/app/utils/preprocess.py
# -*- coding: utf-8 -*-
"""
Módulo de pré-processamento usado em treino e produção.
Define:
 - feature_pipeline: pipeline completo de features para matching
 - preprocessar_candidato: limpeza e formatação de texto de candidato
 - preprocessar_vaga: limpeza e formatação de texto de vaga
"""
import joblib
from ml_api.app.config import Config

# Carrega pipeline completo de pré-processamento (matching)
try:
    feature_pipeline = joblib.load(Config.PIPELINE_PATH)
    print('[INFO] Pipeline de pré-processamento carregado:', Config.PIPELINE_PATH)
except Exception as e:
    print(f'[ERRO] Falha ao carregar pipeline de pré-processamento: {e}')
    raise


def preprocessar_candidato(data: dict) -> str:
    """
    Recebe dicionário com campos brutos de candidato e retorna texto classificado.

    Espera data com:
      - cv_pt: str (texto do currículo)
      - informacoes_profissionais: dict (histórico profissional)
      - formacao_e_idiomas: dict (formação e idiomas)
    """
    # TODO: aplicar limpeza, normalização e concatenação igual ao treino.
    # Por enquanto, retorna apenas o texto bruto do CV.
    return str(data.get('cv_pt', ''))


def preprocessar_vaga(data: dict) -> str:
    """
    Recebe dicionário com campos brutos de vaga e retorna texto processado.

    Espera data com:
      - titulo: str
      - cliente: str
      - requisitos: str (descrição + competências)
    """
    # TODO: aplicar limpeza, normalização e concatenação igual ao treino.
    # Por enquanto, retorna apenas os requisitos concatenados.
    return str(data.get('requisitos', ''))
