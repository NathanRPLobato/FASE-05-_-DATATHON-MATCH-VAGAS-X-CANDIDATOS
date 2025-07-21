import pandas as pd
from ml_api.app.utils.preprocess import extract_tech_features

def test_extract_tech_features_detects_python_and_sql():
    df = pd.DataFrame({
        'texto': ['Tenho experiência em Python e SQL', 'Nenhuma tech aqui']
    })
    out = extract_tech_features(df, 'texto', 'cand')
    # Deve criar colunas cand_tec_python e cand_tec_sql
    assert 'cand_tec_python' in out.columns
    assert 'cand_tec_sql' in out.columns
    assert out.loc[0, 'cand_tec_python'] == 1
    assert out.loc[0, 'cand_tec_sql'] == 1
    assert out.loc[1, 'cand_tec_python'] == 0
    assert out.loc[1, 'cand_tec_sql'] == 0
