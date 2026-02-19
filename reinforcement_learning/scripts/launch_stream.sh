#!/bin/bash
N_ENVS=${1:-12}
BASE_DISPLAY=100
VNC_BASE_PORT=5900
WS_BASE_PORT=6080

echo "Nettoyage..."
killall x11vnc 2>/dev/null
killall websockify 2>/dev/null
sleep 1

echo "Lancement des flux VNC et WebSockets sur 0.0.0.0..."
for (( i=0; i<$N_ENVS; i++ ))
do
    DISPLAY_ID=$((BASE_DISPLAY + i))
    VNC_PORT=$((VNC_BASE_PORT + i))
    WS_PORT=$((WS_BASE_PORT + i))
    
    x11vnc -display :$DISPLAY_ID -bg -nopw -listen localhost -xkb -forever -rfbport $VNC_PORT -quiet
    websockify 0.0.0.0:$WS_PORT localhost:$VNC_PORT -D > /dev/null 2>&1
    echo "Flux Agent $i exposé sur le port $WS_PORT"
done