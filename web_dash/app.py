# web_dash/app.py

from flask import Flask, render_template, request, redirect, url_for
import requests

API_BASE = 'http://localhost:5000'  # API continua na porta 5000

app = Flask(__name__)

@app.route('/')
def index():
    try:
        resp = requests.get(f"{API_BASE}/vagas")
        resp.raise_for_status()
        vagas = resp.json()
    except Exception as e:
        print(f"[ERRO] Falha ao buscar vagas: {e}")
        vagas = []
    return render_template('layout.html', vagas=vagas)

@app.route('/cadastrar_candidato', methods=['GET', 'POST'])
def cadastrar_candidato():
    if request.method == 'POST':
        payload = {
            'nome': request.form['nome'],
            'email': request.form['email'],
            'cv_pt': request.form['cv_pt'],
            'informacoes_profissionais': {},
            'formacao_e_idiomas': {},
            'eh_sap': int('sap' in request.form['cv_pt'].lower())
        }
        r = requests.post(f"{API_BASE}/candidatos", json=payload)
        if r.ok:
            print('Candidato cadastrado com sucesso!')
            return redirect(url_for('index'))
        else:
            print('Falha ao cadastrar candidato:', r.json().get('error'))
    return render_template('register_candidate.html')

@app.route('/cadastrar_vaga', methods=['GET', 'POST'])
def cadastrar_vaga():
    if request.method == 'POST':
        payload = {
            'titulo': request.form['titulo'],
            'cliente': request.form['cliente'],
            'descricao': request.form['descricao'],
            'competencias': request.form['competencias'],
            'eh_sap': int('sap' in request.form['descricao'].lower())
        }
        r = requests.post(f"{API_BASE}/vagas", json=payload)
        if r.ok:
            print('Vaga cadastrada com sucesso!')
            return redirect(url_for('index'))
        else:
            print('Falha ao cadastrar vaga:', r.json().get('error'))
    return render_template('register_vaga.html')

@app.route('/match/<int:vaga_id>')
def match(vaga_id):
    r = requests.get(f"{API_BASE}/match/{vaga_id}")
    if r.ok:
        ranking = r.json()
        return render_template('match.html', ranking=ranking)
    else:
        print('Erro ao gerar match:', r.json().get('error'))
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Dash executa na porta 5001 para simular front-end separado
    app.run(debug=True, port=5001)
