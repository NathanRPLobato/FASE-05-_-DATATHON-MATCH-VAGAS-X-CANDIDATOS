from flask import Flask
from ml_api.app.config import Config
from ml_api.app.routes import register_routes


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    print('[INFO] Configurações carregadas - SQL:', app.config['SQLITE_PATH'])

    register_routes(app)
    print('[INFO] Rotas registradas com sucesso')
    return app