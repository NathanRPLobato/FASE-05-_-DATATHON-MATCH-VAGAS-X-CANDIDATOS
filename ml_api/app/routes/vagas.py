from flask import Blueprint, request, jsonify
from ml_api.app.services.db import inserir_vaga, buscar_vagas
from ml_api.app.services.clustering import cluster_vaga
from ml_api.app.utils.preprocess import preprocessar_vaga
import traceback

vagas_bp = Blueprint('vagas', __name__)

@vagas_bp.route('/', methods=['POST'])
def cria_vaga():
    try:
        data = request.get_json()
        print(f"[INFO] Recebendo vaga (antes de atribuir ID): {data}")

        # 1) Gera novo ID
        existing = buscar_vagas()
        max_id = max((v['id'] for v in existing), default=0)
        new_id = max_id + 1

        # 2) Monta requisitos e processa texto igual ao treino
        requisitos = f"{data['descricao']} {data.get('competencias','')}".strip()
        texto_processado = preprocessar_vaga({
            "titulo": data["titulo"],
            "cliente": data["cliente"],
            "requisitos": requisitos
        })
        print(f"[DEBUG] Texto processado (vaga): {texto_processado}")

        # 3) Clusteriza
        features = {
            "texto_processado": texto_processado,
            "eh_sap": int(data.get("eh_sap", 0))
        }
        cluster = cluster_vaga(features)
        print(f"[DEBUG] cluster_vaga -> {cluster}")

        # 4) Persiste no banco, incluindo cliente e texto_processado
        inserir_vaga({
            "id":        new_id,
            "titulo":    data["titulo"],
            "cliente":   data["cliente"],
            "descricao": texto_processado,
            "cluster":   cluster,
            "eh_sap":    features["eh_sap"]
        })
        print(f"[INFO] Vaga criada com ID={new_id}")
        return jsonify({"status":"ok","id":new_id,"cluster":cluster}), 201

    except KeyError as ke:
        msg = f"campo obrigatório ausente: {ke}"
        print(f"[ERRO] /vagas POST KeyError: {msg}")
        return jsonify({"error": msg}), 400

    except Exception as e:
        traceback.print_exc()
        print(f"[ERRO] /vagas POST Exception: {e}")
        return jsonify({"error": str(e)}), 500

@vagas_bp.route('/', methods=['GET'])
def lista_vagas():
    try:
        return jsonify(buscar_vagas()), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
