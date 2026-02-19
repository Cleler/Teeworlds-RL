#!/usr/bin/env python3
"""
Sert les fichiers statiques noVNC + expose /config.json
généré depuis default.yaml.
"""
import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer

import yaml

YAML_PATH = os.environ.get(
    "CONFIG_PATH",
    "/app/configs/default.yaml"
)
PORT = int(os.environ.get("PORT", 8080))


def load_config() -> dict:
    try:
        with open(YAML_PATH) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[serve] Impossible de lire {YAML_PATH}: {e}")
        return {}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/config.json":
            cfg = load_config()
            multi  = cfg.get("multi_env", {})
            trainer = cfg.get("trainer",    {})

            payload = {
                "n_envs":   multi.get("n_envs",    12),
                "cols":     multi.get("grid_cols",  4),
                "host":     trainer.get("host", "127.0.0.1"),
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type",  "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        pass  # silence les logs de requêtes


if __name__ == "__main__":
    os.chdir("/app/noVNC")          # répertoire de base pour les fichiers statiques
    print(f"[serve] http://0.0.0.0:{PORT}  (config: {YAML_PATH})")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()