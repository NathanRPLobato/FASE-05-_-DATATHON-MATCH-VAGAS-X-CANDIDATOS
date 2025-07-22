import os
import requests
from flask import Flask, render_template, request, redirect, url_for

# Define a URL base da API de Consumo
API_BASE = os.getenv("API_BASE", "http://localhost:5000")

app = Flask(__name__)


@app.route('/')
def index():
    """Lista todas as vagas cadastradas na API."""
    try:
        resp = requests.get(f"{API_BASE}/vagas/")
        resp.raise_for_status()
        vagas = resp.json()
    except Exception as e:
        app.logger.error(f"Falha ao buscar vagas: {e}")
        vagas = []
    return render_template('layout.html', vagas=vagas)


@app.route('/cadastrar_candidato', methods=['GET', 'POST'])
def cadastrar_candidato():
    """Formulário e submissão de novo candidato."""
    if request.method == 'POST':
        payload = {
            "nome": request.form['nome'],
            "email": request.form['email'],
            "cv_pt": request.form['cv_pt'],
            "informacoes_profissionais": {},
            "formacao_e_idiomas": {},
            "eh_sap": int('sap' in request.form['cv_pt'].lower())
        }
        r = requests.post(f"{API_BASE}/candidatos/", json=payload)
        if r.ok:
            return redirect(url_for('index'))
        else:
            app.logger.error(f"Falha ao cadastrar candidato: {r.text}")
    return render_template('register_candidate.html')


@app.route('/cadastrar_vaga', methods=['GET', 'POST'])
def cadastrar_vaga():
    """Formulário e submissão de nova vaga."""
    if request.method == 'POST':
        payload = {
            "titulo": request.form['titulo'],
            "cliente": request.form['cliente'],
            "descricao": request.form['descricao'],
            "competencias": request.form['competencias'],
            "eh_sap": int('sap' in request.form['descricao'].lower())
        }
        r = requests.post(f"{API_BASE}/vagas/", json=payload)
        if r.ok:
            return redirect(url_for('index'))
        else:
            app.logger.error(f"Falha ao cadastrar vaga: {r.text}")
    return render_template('register_vaga.html')


@app.route('/match/<int:vaga_id>')
def match(vaga_id):
    """Gera e exibe ranking top‑10 candidatos para a vaga selecionada."""
    try:
        r = requests.get(f"{API_BASE}/match/{vaga_id}")
        r.raise_for_status()
        ranking = r.json()
        return render_template('match.html', ranking=ranking)
    except Exception as e:
        app.logger.error(f"Erro ao gerar match para vaga {vaga_id}: {e}")
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
