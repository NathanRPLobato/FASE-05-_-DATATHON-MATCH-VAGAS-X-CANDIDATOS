import sqlite3
import tempfile
from ml_api.app.services.db import inserir_candidato, buscar_candidatos

def test_db_insert_and_query():
    db = sqlite3.connect(':memory:')
    # criar tabela identica...
    db.execute("""
      CREATE TABLE candidatos (
        id INTEGER PRIMARY KEY,
        nome TEXT, email TEXT, texto_classificado TEXT, cluster INTEGER
      )
    """)
    inserir_candidato({'id':1,'nome':'X','email':'x@x','texto_classificado':'X','cluster':0}, conn=db)
    rows = buscar_candidatos(conn=db)
    assert len(rows) == 1
    assert rows[0]['nome'] == 'X'
