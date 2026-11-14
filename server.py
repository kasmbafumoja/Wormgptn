import os
from flask import Flask, request, jsonify
# On importe ton application existante
from api.index import app as wsgi_app

app = Flask(__name__)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(path):
    # On prépare l'environnement pour ton code WSGI
    environ = request.environ.copy()
    environ['PATH_INFO'] = '/' + path
    environ['REQUEST_METHOD'] = request.method
    environ['wsgi.input'] = request.stream
    if request.content_length:
        environ['CONTENT_LENGTH'] = str(request.content_length)

    # On récupère la réponse de ton code
    def start_response(status, headers):
        pass
    
    response_body = wsgi_app(environ, start_response)

    # On renvoie la réponse via Flask
    status_code = 200
    if isinstance(response_body, list) and len(response_body) > 0:
        return response_body[0], status_code, {'Content-Type': 'application/json'}
    return jsonify({"error": "Empty response"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)