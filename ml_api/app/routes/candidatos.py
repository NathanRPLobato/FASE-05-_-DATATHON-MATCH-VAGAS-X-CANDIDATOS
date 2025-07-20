from flask import Blueprint, request, jsonify
from ml_api.app.services.db import inserir_candidato, buscar_candidatos
from ml_api.app.services.clustering import cluster_candidato
from ml_api.app.utils.preprocess import preprocessar_candidato
import traceback

candidatos_bp = Blueprint('candidatos', __name__)

@candidatos_bp.route('/', methods=['POST'])
def cria_candidato():
    try:
        data = request.get_json()
        print(f"[INFO] Recebendo candidato (antes de atribuir ID): {data}")

        # 1) Gera novo ID: maior ID existente + 1
        cand_existentes = buscar_candidatos()
        max_id = max((c['id'] for c in cand_existentes), default=0)
        new_id = max_id + 1

        # 2) Pré-processamento idêntico ao treino
        processed_text = preprocessar_candidato({
            'cv_pt': data['cv_pt'],
            'informacoes_profissionais': data.get('informacoes_profissionais', {}),
            'formacao_e_idiomas': data.get('formacao_e_idiomas', {})
        })
        print(f"[DEBUG] Texto classificado (candidato): {processed_text}")

        # 3) Monta features para clustering
        features = {
            'texto_classificado': processed_text,
            'eh_sap':             int(data.get('eh_sap', 0))
        }
        print(f"[DEBUG] Features candidato: {features}")

        # 4) Aplica cluster
        cluster = cluster_candidato(features)
        print(f"[DEBUG] cluster_candidato -> {cluster}")

        # 5) Persiste no DB (incluindo texto_classificado)
        inserir_candidato({
            'id':                new_id,
            'nome':              data.get('nome', ''),
            'email':             data.get('email', ''),
            'texto_classificado': processed_text,
            'cluster':           cluster,
            'eh_sap':            features['eh_sap']
        })
        print(f"[INFO] Candidato criado com ID={new_id}")
        return jsonify({'status':'ok','id': new_id,'cluster': cluster}), 201

    except KeyError as ke:
        msg = f"campo obrigatório ausente: {ke}"
        print(f"[ERRO] /candidatos POST KeyError: {msg}")
        return jsonify({'error': msg}), 400

    except Exception as e:
        traceback.print_exc()
        print(f"[ERRO] /candidatos POST Exception: {e}")
        return jsonify({'error': str(e)}), 500

@candidatos_bp.route('/', methods=['GET'])
def lista_candidatos():
    try:
        return jsonify(buscar_candidatos()), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500