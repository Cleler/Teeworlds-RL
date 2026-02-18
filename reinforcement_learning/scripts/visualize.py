#!/usr/bin/env python3
import time
from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# On stocke les images en mémoire (Dictionnaire: Bot_ID -> Bytes JPEG)
LATEST_FRAMES = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Teeworlds RL - Visualizer Distant</title>
    <style>
        body { background-color: #1e1e1e; color: #fff; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
        .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 20px; }
        .bot-card { background-color: #2d2d2d; border: 2px solid #444; border-radius: 10px; padding: 15px; }
        .bot-card h3 { margin: 0 0 10px 0; color: #4CAF50; }
        img { width: 252px; height: 252px; image-rendering: pixelated; border-radius: 4px; background-color: #000; }
    </style>
    <script>
        if ({{ bot_ids|length }} === 0) {
            setTimeout(function() { location.reload(); }, 2000);
        }
    </script>
</head>
<body>
    <h1>Teeworlds RL - Live Vision (Réseau) 📡</h1>
    {% if bot_ids|length == 0 %}
        <h2 style="color: #888;">En attente de la connexion du train.py distant...</h2>
    {% else %}
        <div class="grid">
            {% for bot_id in bot_ids %}
            <div class="bot-card">
                <h3>Agent {{ bot_id }}</h3>
                <img src="{{ url_for('video_feed', bot_id=bot_id) }}" alt="Stream Bot {{ bot_id }}">
            </div>
            {% endfor %}
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    bot_ids = sorted(list(LATEST_FRAMES.keys()))
    return render_template_string(HTML_TEMPLATE, bot_ids=bot_ids)

# --- NOUVELLE ROUTE : Pour recevoir les images de train.py ---
@app.route('/update/<int:bot_id>', methods=['POST'])
def update_frame(bot_id):
    # On met à jour l'image en mémoire avec les données brutes reçues
    LATEST_FRAMES[bot_id] = request.data
    return "OK", 200

def generate_video_stream(bot_id):
    while True:
        frame_bytes = LATEST_FRAMES.get(bot_id)
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/bot/<int:bot_id>')
def video_feed(bot_id):
    return Response(generate_video_stream(bot_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🌐 Visualizer distant en écoute sur le port 5000...")
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)