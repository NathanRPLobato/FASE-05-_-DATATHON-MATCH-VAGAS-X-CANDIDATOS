from flask import Blueprint, jsonify
from ml_api.app.services.db import buscar_candidatos, buscar_vagas
from ml_api.app.services.pipeline_match import ranquear_candidatos_para_vaga

match_bp = Blueprint('match', __name__)

@match_bp.route('/<int:vaga_id>', methods=['GET'])
def match_vaga(vaga_id):
    try:
        print(f"[INFO] Match para vaga ID={vaga_id}...")
        vaga = next((v for v in buscar_vagas() if v['id']==vaga_id), None)
        if not vaga:
            return jsonify({'error':f'vaga {vaga_id} não encontrada'}), 404

        ranking = ranquear_candidatos_para_vaga(vaga, buscar_candidatos())
        return jsonify(ranking), 200

    except Exception as e:
        return jsonify({'error':str(e)}), 500