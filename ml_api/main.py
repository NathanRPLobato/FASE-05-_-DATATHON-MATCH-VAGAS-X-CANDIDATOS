# ml_api/main.py

import os
from flask import Flask, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
from ml_api.app import create_app

# Cria a app principal
app = create_app()

# --- Rota para servir o seu swagger_template.yml que está em ml_api/app/ ---
@app.route('/swagger_template.yml')
def swagger_yaml():
    # app.root_path -> .../ml_api/app
    return send_from_directory(app.root_path, 'swagger_template.yml')

# --- Swagger UI setup ---
SWAGGER_URL = '/docs'                 # URL da interface
API_URL     = '/swagger_template.yml' # local onde o YAML é servido

swaggerui_bp = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Match Vagas API"}
)
app.register_blueprint(swaggerui_bp, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    # Porta 5000 para a API
    app.run(debug=True, host='0.0.0.0', port=5000)
