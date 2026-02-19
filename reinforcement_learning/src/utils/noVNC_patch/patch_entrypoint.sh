#!/bin/sh
set -e

if [ -f /app/patch/novnc.patch ]; then
    if [ ! -f /app/.patches_applied ]; then
        echo "Applying patches..."
        
        patch -p1 -N < /app/patch/novnc.patch
        mv /app/patch/dashboard.html /app/noVNC

        touch /app/.patches_applied
        echo "Patches applied successfully."
    else
        echo "Patches already applied, skipping..."
    fi
else
    echo "No patch files found, skipping..."
fi

pip install pyyaml --quiet

exec python /app/serve.py