import os
import re
import json
import requests
from datetime import datetime

try:
    from langdetect import detect
except ImportError:
    detect = None

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, '..', 'wormgpt_config.json')
PROMPT_PATH = os.path.join(BASE_DIR, '..', 'prompt.txt')

DEFAULT_CONFIG = {
    'default_provider': 'gemini',
    'language': 'Français',
    'site_url': 'https://github.com/00x0kafyy/worm-ai',
    'site_name': 'WormGPT API',
    'routes': {'root': True, 'chat': True, 'health': True},
    'providers': {
        'gemini': {
            'enabled': True,
            'api_key_env': 'GEMINI_API_KEY',
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        }
    },
    'features': {'enable_history': True, 'history_limit': 16},
    'defaults': {'temperature': 0.7, 'max_tokens': 1400},
    'model': {'name': 'gemini-1.5-flash'},
}


def app(environ, start_response):
    method = environ.get('REQUEST_METHOD', 'GET').upper()
    path = environ.get('PATH_INFO', '/')

    if method == 'OPTIONS':
        return json_response(start_response, 200, {'status': 'success'})

    try:
        config = load_config()

        if path == '/' and method == 'GET':
            return json_response(start_response, 200, {
                'status': 'success',
                'message': 'WormGPT API (Gemini) is running',
                'version': '6.0-chat-only',
                'model': config['model']['name'],
                'routes': {k: bool(v) for k, v in config.get('routes', {}).items()},
                'timestamp': utc_now(),
            })

        if path == '/api/health' and method == 'GET':
            return json_response(start_response, 200, {
                'status': 'success',
                'service': 'healthy',
                'has_gemini_key': bool(resolve_gemini_api_key(config).strip()),
                'has_prompt_file': os.path.exists(PROMPT_PATH),
                'timestamp': utc_now(),
            })

        if path == '/api/chat' and method == 'POST':
            return handle_chat(start_response, config, read_json_body(environ))

        return json_response(start_response, 404, {'status': 'error', 'message': 'Route introuvable'})
    except Exception as e:
        return json_response(start_response, 500, {'status': 'error', 'message': str(e)})


def handle_chat(start_response, config, body):
    user_input = (body.get('message') or '').strip()
    history = sanitize_history(body.get('history') or [], config)
    language = (body.get('language') or config.get('language') or 'Français').strip()
    inferred_modality = detect_input_modality(body)

    if not user_input and not body.get('messages') and inferred_modality == 'text':
        return json_response(start_response, 400, {'status': 'error', 'message': "Le champ 'message' est requis."})

    api_key = resolve_gemini_api_key(config)
    if not api_key:
        return json_response(start_response, 400, {'status': 'error', 'message': 'Clé API manquante. Ajoute GEMINI_API_KEY dans Vercel.'})

    system_prompt = get_system_prompt()
    if system_prompt is None:
        return json_response(start_response, 500, {
            'status': 'error',
            'message': "Fichier prompt.txt introuvable ou vide. Crée-le à la racine du projet.",
        })

    if body.get('messages'):
        # Le client impose ses messages : on filtre tout 'system' et on force le nôtre
        client_msgs = [
            m for m in body['messages']
            if isinstance(m, dict) and m.get('role') != 'system'
        ]
        messages = [{'role': 'system', 'content': system_prompt}] + client_msgs
    else:
        messages = build_messages(config, body, system_prompt, history, user_input, inferred_modality)

    detected_language = normalize_language(language, user_input)

    result = call_gemini(
        config, api_key, messages,
        temperature=body.get('temperature', config.get('defaults', {}).get('temperature', 0.7)),
        max_tokens=body.get('max_tokens', config.get('defaults', {}).get('max_tokens', 1400)),
    )
    if result.get('status') != 'success':
        return json_response(start_response, 200, result)

    return json_response(start_response, 200, {
        'status': 'success',
        'message': user_input,
        'response': result.get('text', ''),
        'language': detected_language,
        'model': config['model']['name'],
        'provider': 'gemini',
        'modality_detected': inferred_modality,
        'timestamp': utc_now(),
    })


# ---------------------------------------------------------------------------
# Appel Gemini (format natif)
# ---------------------------------------------------------------------------

def call_gemini(config, api_key, messages, temperature=0.7, max_tokens=1400):
    base_url = config.get('providers', {}).get('gemini', {}).get(
        'base_url', 'https://generativelanguage.googleapis.com/v1beta').rstrip('/')
    model = config['model']['name']
    url = f'{base_url}/models/{model}:generateContent?key={api_key}'

    system_text_parts = []
    contents = []

    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')

        if role == 'system':
            system_text_parts.append(extract_text(content))
            continue

        gemini_role = 'model' if role == 'assistant' else 'user'
        parts = to_gemini_parts(content)
        if not parts:
            continue
        contents.append({'role': gemini_role, 'parts': parts})

    payload = {
        'contents': contents,
        'generationConfig': {
            'temperature': temperature,
            'maxOutputTokens': max_tokens,
        },
    }
    if system_text_parts:
        payload['systemInstruction'] = {'parts': [{'text': '\n\n'.join(system_text_parts)}]}

    try:
        response = requests.post(url, json=payload, timeout=90)
        if not response.ok:
            return {'status': 'error', 'message': f'[Gemini] API Error {response.status_code}: {response.text}'}
        return parse_gemini_response(response.json())
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': '[Gemini] API Error: délai dépassé.'}
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'message': f'[Gemini] API Error: {str(e)}'}
    except Exception as e:
        return {'status': 'error', 'message': f'[Gemini] API Error: {str(e)}'}


def parse_gemini_response(result):
    text_parts = []
    for cand in result.get('candidates', []) or []:
        for part in cand.get('content', {}).get('parts', []) or []:
            if part.get('text'):
                text_parts.append(part['text'])
    return {'status': 'success', 'text': '\n'.join(text_parts).strip()}


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def to_gemini_parts(content):
    if isinstance(content, str):
        return [{'text': content}] if content else []

    parts = []
    if not isinstance(content, list):
        return parts

    for item in content:
        if not isinstance(item, dict):
            continue
        ptype = item.get('type')

        if ptype == 'text' and item.get('text'):
            parts.append({'text': item['text']})

        elif ptype == 'image_url':
            url = (item.get('image_url') or {}).get('url') or item.get('url')
            inline = data_url_to_inline(url)
            if inline:
                parts.append({'inlineData': inline})

        elif ptype in ('input_audio', 'audio') and item.get('input_audio'):
            audio = item['input_audio']
            fmt = normalize_mime_audio(audio.get('format', 'mp3'))
            if audio.get('data'):
                parts.append({'inlineData': {'mimeType': fmt, 'data': audio.get('data')}})

        elif ptype == 'file':
            f = item.get('file') or {}
            data_url = f.get('file_data') or f.get('file_url')
            inline = data_url_to_inline(data_url)
            if inline:
                parts.append({'inlineData': inline})

        elif ptype == 'video_url':
            url = (item.get('video_url') or {}).get('url') or item.get('url')
            inline = data_url_to_inline(url)
            if inline:
                parts.append({'inlineData': inline})

    return parts


def data_url_to_inline(url):
    if not url:
        return None
    m = re.match(r'^data:([^;]+);base64,(.+)$', url, re.I | re.S)
    if m:
        return {'mimeType': m.group(1), 'data': m.group(2)}
    return None


def normalize_mime_audio(fmt):
    fmt = (fmt or 'mp3').lower()
    mapping = {'mp3': 'audio/mp3', 'wav': 'audio/wav', 'mpeg': 'audio/mpeg', 'ogg': 'audio/ogg', 'webm': 'audio/webm'}
    return mapping.get(fmt, f'audio/{fmt}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        for key in ('routes', 'providers', 'features', 'defaults'):
            merged[key] = {**DEFAULT_CONFIG.get(key, {}), **data.get(key, {})}
        if 'model' not in merged or not merged['model'].get('name'):
            merged['model'] = DEFAULT_CONFIG['model']
        return merged
    except Exception:
        return DEFAULT_CONFIG


def read_json_body(environ):
    try:
        content_length = int(environ.get('CONTENT_LENGTH', '0') or '0')
    except ValueError:
        content_length = 0
    body = environ['wsgi.input'].read(content_length) if content_length > 0 else b'{}'
    if not body:
        return {}
    try:
        return json.loads(body.decode('utf-8'))
    except Exception:
        return {}


def json_response(start_response, status_code, data):
    status_map = {
        200: '200 OK', 400: '400 Bad Request', 404: '404 Not Found',
        405: '405 Method Not Allowed', 500: '500 Internal Server Error',
    }
    status_text = status_map.get(status_code, f'{status_code} OK')
    response_body = json.dumps(data, ensure_ascii=False).encode('utf-8')
    headers = [
        ('Content-Type', 'application/json; charset=utf-8'),
        ('Content-Length', str(len(response_body))),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
        ('Access-Control-Allow-Headers', 'Content-Type, Authorization'),
    ]
    start_response(status_text, headers)
    return [response_body]


def utc_now():
    return datetime.utcnow().isoformat() + 'Z'


def resolve_gemini_api_key(config):
    provider = config.get('providers', {}).get('gemini', {})
    raw = str(provider.get('api_key_env', 'GEMINI_API_KEY')).strip()
    if raw.startswith('AIza'):
        return raw
    if raw and raw != 'GEMINI_API_KEY':
        env_value = os.getenv(raw, '').strip()
        if env_value:
            return env_value
    return os.getenv('GEMINI_API_KEY', '').strip()


def get_system_prompt():
    """
    Lit le prompt système UNIQUEMENT depuis prompt.txt.
    Retourne None si le fichier est absent ou vide.
    """
    if not os.path.exists(PROMPT_PATH):
        return None
    try:
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None
    except Exception:
        return None


def normalize_language(current_language, user_input):
    if detect is None or not user_input:
        return current_language
    try:
        detected = detect(user_input[:500])
        lang_map = {
            'en': 'English', 'fr': 'Français', 'es': 'Spanish',
            'ar': 'Arabic', 'th': 'Thai', 'pt': 'Portuguese',
        }
        return lang_map.get(detected, current_language)
    except Exception:
        return current_language


def sanitize_history(history, config):
    if not isinstance(history, list):
        return []
    limit = int(config.get('features', {}).get('history_limit', 16))
    clean = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        content = item.get('content')
        if role not in ('user', 'assistant', 'system') or content is None:
            continue
        clean.append({'role': role, 'content': content})
    return clean


def detect_input_modality(body):
    if body.get('video_url') or body.get('video_data_url'):
        return 'video'
    if body.get('audio_url') or body.get('audio_data_url') or body.get('audio_base64'):
        return 'audio'
    if body.get('pdf_url') or body.get('pdf_data_url'):
        return 'pdf'
    if body.get('image_url') or body.get('image_data_url'):
        return 'image'
    return 'text'


def build_messages(config, body, system_prompt, history, user_input, inferred_modality):
    messages = [{'role': 'system', 'content': system_prompt}]
    if config.get('features', {}).get('enable_history', True):
        messages.extend(history)

    if inferred_modality == 'text':
        messages.append({'role': 'user', 'content': user_input})
        return messages

    content = [{'type': 'text', 'text': user_input if user_input else 'Analyse ce contenu.'}]
    append_image_inputs(content, body)
    append_pdf_inputs(content, body)
    append_audio_inputs(content, body)
    append_video_inputs(content, body)
    messages.append({'role': 'user', 'content': content})
    return messages


def append_image_inputs(content, body):
    if body.get('image_url'):
        content.append({'type': 'image_url', 'image_url': {'url': body.get('image_url')}})
    if body.get('image_data_url'):
        content.append({'type': 'image_url', 'image_url': {'url': body.get('image_data_url')}})


def append_pdf_inputs(content, body):
    if body.get('pdf_url'):
        content.append({'type': 'file', 'file': {'filename': body.get('pdf_filename', 'document.pdf'), 'file_url': body.get('pdf_url')}})
    if body.get('pdf_data_url'):
        content.append({'type': 'file', 'file': {'filename': body.get('pdf_filename', 'document.pdf'), 'file_data': body.get('pdf_data_url')}})


def append_audio_inputs(content, body):
    audio_data_url = body.get('audio_data_url')
    if audio_data_url:
        m = re.match(r'^data:audio\/([a-zA-Z0-9.+-]+);base64,(.+)$', audio_data_url, re.I | re.S)
        if m:
            content.append({'type': 'input_audio', 'input_audio': {'data': m.group(2), 'format': m.group(1)}})
            return
    if body.get('audio_base64'):
        content.append({'type': 'input_audio', 'input_audio': {'data': body.get('audio_base64'), 'format': body.get('audio_format') or 'mp3'}})
    elif body.get('audio_url'):
        content.append({'type': 'text', 'text': f"Audio URL: {body.get('audio_url')}"})


def append_video_inputs(content, body):
    if body.get('video_url'):
        content.append({'type': 'video_url', 'video_url': {'url': body.get('video_url')}})
    if body.get('video_data_url'):
        content.append({'type': 'video_url', 'video_url': {'url': body.get('video_data_url')}})
