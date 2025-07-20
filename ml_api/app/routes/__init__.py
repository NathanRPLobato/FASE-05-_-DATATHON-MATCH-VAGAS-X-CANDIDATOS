from ml_api.app.routes.candidatos import candidatos_bp
from ml_api.app.routes.vagas import vagas_bp
from ml_api.app.routes.match import match_bp


def register_routes(app):
    app.register_blueprint(candidatos_bp, url_prefix='/candidatos')
    app.register_blueprint(vagas_bp,      url_prefix='/vagas')
    app.register_blueprint(match_bp,      url_prefix='/match')