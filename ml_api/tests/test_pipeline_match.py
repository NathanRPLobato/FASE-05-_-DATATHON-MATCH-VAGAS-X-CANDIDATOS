from ml_api.app.services.pipeline_match import ranquear_candidatos_para_vaga

def test_ranking_empty_candidates():
    vaga = {'cluster': 0, 'descricao': 'Teste', 'eh_sap': 0}
    ranking = ranquear_candidatos_para_vaga(vaga, [])
    assert ranking == []

def test_ranking_structure():
    vaga = {'cluster': 0, 'descricao': 'Python SQL', 'eh_sap': 0}
    # candidado mínimo para passar pelas checks:
    candidatos = [{
        'id': 1, 'nome': 'A', 'email':'a@x','cluster':0,
        'eh_sap':0,'texto_classificado':'Python'
    }]
    ranking = ranquear_candidatos_para_vaga(vaga, candidatos)
    assert isinstance(ranking, list)
    assert ranking[0]['id'] == 1
    assert 'compatibilidade' in ranking[0]
