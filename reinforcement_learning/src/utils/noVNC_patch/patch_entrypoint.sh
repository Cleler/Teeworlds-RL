#!/bin/sh
set -e

if [ -f /app/patch/novnc_dashboard.patch ] && [ -f /app/patch/novnc.patch ]; then
    echo "Applying patch..."
    patch -p1 < /app/patch/novnc_dashboard.patch
    patch -p1 < /app/patch/novnc.patch
else
    echo "No patch found, skipping..."
fi

exec python -m http.server 8080
