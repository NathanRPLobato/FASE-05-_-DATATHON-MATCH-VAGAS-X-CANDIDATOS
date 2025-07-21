from ml_api.app.services.clustering import cluster_vaga, cluster_candidato

def test_cluster_vaga_returns_int():
    dummy = {'titulo': 'Engenheiro', 'descricao': 'Java e AWS', 'eh_sap': 0}
    c = cluster_vaga(dummy)
    assert isinstance(c, int)

def test_cluster_candidato_returns_int():
    dummy = {'texto_classificado': 'Python e SQL', 'eh_sap': 1}
    c = cluster_candidato(dummy)
    assert isinstance(c, int)
