import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Ajusta PYTHONPATH para que import ml_api funcione
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_api.app.config import DB_PATH

# Caminhos para os CSVs refined
CAND_CSV = ROOT / "ml_api" / "data" / "refined" / "candidatos.csv"
VAGA_CSV = ROOT / "ml_api" / "data" / "refined" / "vagas.csv"


def init_db(conn: sqlite3.Connection):
    """cria as tabelas esperadas pela API: candidatos e vagas."""
    c = conn.cursor()
    # Tabela candidatos
    c.execute("DROP TABLE IF EXISTS candidatos;")
    c.execute("""
    CREATE TABLE candidatos (
        id INTEGER PRIMARY KEY,
        nome TEXT NOT NULL,
        email TEXT NOT NULL,
        texto_classificado TEXT NOT NULL,
        cluster INTEGER NOT NULL,
        eh_sap INTEGER NOT NULL
    );
    """)
    # Tabela vagas
    c.execute("DROP TABLE IF EXISTS vagas;")
    c.execute("""
    CREATE TABLE vagas (
        id INTEGER PRIMARY KEY,
        titulo TEXT NOT NULL,
        descricao TEXT NOT NULL,
        cluster INTEGER NOT NULL,
        eh_sap INTEGER NOT NULL
    );
    """)
    conn.commit()


def load_and_insert_candidatos(conn: sqlite3.Connection):
    """Carrega candidatos.csv e insere na tabela candidatos."""
    df = pd.read_csv(CAND_CSV, dtype={"id": int, "cluster": int})
    # Calcula flag SAP a partir do texto_classificado
    df['eh_sap'] = (
        df['texto_classificado']
        .astype(str)
        .str.lower()
        .str.contains('sap', na=False)
        .astype(int)
    )
    cols = ["id", "nome", "email", "texto_classificado", "cluster", "eh_sap"]
    df[cols].to_sql("candidatos", conn, if_exists="append", index=False)


def load_and_insert_vagas(conn: sqlite3.Connection):
    """Carrega vagas.csv e insere na tabela vagas."""
    df = pd.read_csv(VAGA_CSV, encoding="utf-8-sig")
    # Renomeia colunas para coincidir com o schema
    if 'vaga_id' in df.columns:
        df = df.rename(columns={'vaga_id': 'id'})
    if 'classification' in df.columns:
        df = df.rename(columns={'classification': 'cluster'})
    if 'é_sap' in df.columns:
        df = df.rename(columns={'é_sap': 'eh_sap'})

    # Constrói descricao combinando descricao + competencias
    descricao_col    = df.get('descricao',    pd.Series(['']*len(df), dtype=str))
    competencias_col = df.get('competencias', pd.Series(['']*len(df), dtype=str))
    df['descricao'] = (descricao_col.fillna('') + ' ' + competencias_col.fillna('')).str.strip()

    # Evita duplicatas de ID
    df = df.drop_duplicates(subset=['id'])

    # Converte tipos
    df['eh_sap']  = df.get('eh_sap', 0).astype(int)
    df['cluster'] = df['cluster'].astype(int)

    cols = ['id', 'titulo', 'descricao', 'cluster', 'eh_sap']
    df[cols].to_sql('vagas', conn, if_exists='append', index=False)


def main():
    # Garante existência da pasta do DB
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))

    init_db(conn)
    load_and_insert_candidatos(conn)
    load_and_insert_vagas(conn)

    conn.close()
    print(f"Banco criado e populado em {DB_PATH}")


if __name__ == "__main__":
    main()
