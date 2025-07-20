from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'model'
DB_PATH   = BASE_DIR / 'data' / 'gold' / 'decision_match.db'

class Config:
    # Caminhos para arquivos
    SQLITE_PATH        = str(DB_PATH)
    PIPELINE_PATH      = str(MODEL_DIR / 'feature_pipeline.joblib')
    CAND_CLUSTER_PATH  = str(MODEL_DIR / 'candidato_cluster_model.joblib')
    VAGA_CLUSTER_PATH  = str(MODEL_DIR / 'vaga_cluster_model.joblib')