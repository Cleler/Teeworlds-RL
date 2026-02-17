# Teeworlds RL

Reinforcement Learning agent pour Teeworlds utilisant :
- **Gymnasium** comme interface RL
- **mss** pour la capture d'écran
- **pyautogui** pour l'injection d'inputs
- **econ** (external console) pour la communication avec le serveur TW

## Architecture

```
teeworlds-rl/
├── configs/
│   └── default.yaml            # Configuration (serveur, capture, RL)
├── scripts/
│   ├── train.py                # Point d'entrée entraînement
│   ├── evaluate.py             # Point d'entrée évaluation
│   └── launch_server.sh        # Lancer le serveur TW
├── src/
│   ├── env/
│   │   ├── econ_client.py      # Connexion econ au serveur TW
│   │   ├── screen_capture.py   # Capture d'écran via mss
│   │   ├── input_controller.py # Injection inputs via pyautogui
│   │   └── teeworlds_env.py    # Wrapper Gymnasium
│   ├── agent/
│   │   ├── network.py          # Architecture réseau (CNN + MLP)
│   │   └── train.py            # Logique d'entraînement
│   └── utils/
│       └── config.py           # Chargement config
├── requirements.txt
└── README.md
```

## Flow des données

```
  ┌─────────────────────────────────────────────────────────┐
  │                    Gymnasium Env                         │
  │                  (teeworlds_env.py)                      │
  │                                                         │
  │  ┌───────────────┐  ┌────────────────┐  ┌────────────┐ │
  │  │ ScreenCapture │  │ InputController│  │ EconClient │ │
  │  │  (mss)        │  │ (pyautogui)    │  │ (TCP)      │ │
  │  │               │  │                │  │            │ │
  │  │  obs: image   │  │  act: keys +   │  │ obs: pos,  │ │
  │  │  84x84 gray   │  │  souris        │  │ hp, score  │ │
  │  └───────┬───────┘  └───────▲────────┘  └─────┬──────┘ │
  │          │                  │                  │        │
  └──────────┼──────────────────┼──────────────────┼────────┘
             │                  │                  │
             ▼                  │                  ▼
       ┌─────────┐        ┌────┴─────┐      ┌──────────┐
       │ Client  │        │ Agent RL │      │ Serveur  │
       │ TW GUI  │        │ (PPO)    │      │ TW       │
       └─────────┘        └──────────┘      └──────────┘
```

## Setup

1. Installer les dépendances : `pip install -r requirements.txt`
2. Configurer `configs/default.yaml` (IP serveur, econ password, etc.)
3. Lancer le serveur TW : `bash scripts/launch_server.sh`
4. Lancer le client TW manuellement et rejoindre le serveur
5. Lancer l'entraînement : `python scripts/train.py`

## Configuration serveur TW requise

Dans `autoexec.cfg` du serveur :
```
ec_port 8303
ec_password "password"
ec_output_level 2
sv_max_clients 2
```
