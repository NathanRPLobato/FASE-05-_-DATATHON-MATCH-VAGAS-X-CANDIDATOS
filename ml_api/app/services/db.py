import sqlite3
from ml_api.app.config import Config

#Funções de CRUD para o banco de dados SQLite

def get_connection():
    conn = sqlite3.connect(Config.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection(); cur = conn.cursor()
    print('[INFO] Inicializando banco de dados...')
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS candidatos (
        id INTEGER PRIMARY KEY,
        nome TEXT NOT NULL,
        email TEXT NOT NULL,
        texto_classificado TEXT NOT NULL,
        cluster INTEGER NOT NULL,
        eh_sap INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS vagas (
        id INTEGER PRIMARY KEY,
        titulo TEXT NOT NULL,
        descricao TEXT NOT NULL,
        cluster INTEGER NOT NULL,
        eh_sap INTEGER NOT NULL
    );
    ''')
    conn.commit(); conn.close()
    print('[INFO] Banco de dados pronto')


def inserir_candidato(cand: dict):
    try:
        conn = get_connection(); cur = conn.cursor()
        print(f"[INFO] Inserindo/atualizando candidato ID={cand['id']}")
        cur.execute(
            'REPLACE INTO candidatos (id, nome, email, texto_classificado, cluster, eh_sap) VALUES (?, ?, ?, ?, ?, ?)',
            (
                cand['id'],
                cand['nome'],
                cand['email'],
                cand.get('texto_classificado', ''),
                int(cand['cluster']),
                int(cand['eh_sap'])
            )
        )
        conn.commit()
    except Exception as e:
        print(f"[ERRO] Inserir candidato falhou: {e}")
        raise
    finally:
        conn.close()


def inserir_vaga(vaga: dict):
    try:
        conn = get_connection(); cur = conn.cursor()
        print(f"[INFO] Inserindo/atualizando vaga ID={vaga['id']}")
        cur.execute(
            'REPLACE INTO vagas (id, titulo, descricao, cluster, eh_sap) VALUES (?, ?, ?, ?, ?)',
            (
                vaga['id'],
                vaga['titulo'],
                vaga['descricao'],
                int(vaga['cluster']),
                int(vaga['eh_sap'])
            )
        )
        conn.commit()
    except Exception as e:
        print(f"[ERRO] Inserir vaga falhou: {e}")
        raise
    finally:
        conn.close()


def buscar_candidatos():
    conn = get_connection(); cur = conn.cursor()
    cur.execute('SELECT * FROM candidatos'); rows = cur.fetchall(); conn.close()
    print(f"[INFO] {len(rows)} candidatos encontrados no DB")
    return [dict(r) for r in rows]


def buscar_vagas():
    conn = get_connection(); cur = conn.cursor()
    cur.execute('SELECT * FROM vagas'); rows = cur.fetchall(); conn.close()
    print(f"[INFO] {len(rows)} vagas encontradas no DB")
    return [dict(r) for r in rows]