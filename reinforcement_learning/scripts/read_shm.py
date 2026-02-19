#!/usr/bin/env python3
"""
Lit la shared memory de chaque agent et affiche l'état en temps réel.
Usage : python read_shm.py [n_agents]
"""
import mmap, os, struct, time, sys

FMT  = "=iiiiiffiiQ"
SIZE = struct.calcsize(FMT)
FIELDS = ["dir", "jump", "fire", "hook", "wpn", "aim_x", "aim_y", "win_w", "win_h", "seq"]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4

handles = []
for i in range(n):
    path = f"/dev/shm/tw_input_{i}"
    if not os.path.exists(path):
        print(f"Agent {i}: SHM introuvable ({path})")
        handles.append(None)
        continue
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    handles.append((fd, mm))
    print(f"Agent {i}: SHM OK")

print("\n--- Lecture en temps réel (Ctrl+C pour quitter) ---\n")
prev_seqs = [0] * n

try:
    while True:
        for i, h in enumerate(handles):
            if h is None:
                continue
            fd, mm = h
            mm.seek(0)
            data = struct.unpack(FMT, mm.read(SIZE))
            state = dict(zip(FIELDS, data))
            
            # Afficher seulement si la séquence a changé
            if state["seq"] != prev_seqs[i]:
                prev_seqs[i] = state["seq"]
                print(f"Agent {i:2d} | seq={state['seq']:6d} | "
                      f"dir={state['dir']:+d} | "
                      f"jump={state['jump']} | "
                      f"fire={state['fire']} | "
                      f"hook={state['hook']} | "
                      f"wpn={state['wpn']} | "
                      f"aim=({state['aim_x']:+.2f},{state['aim_y']:+.2f})")
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nFin.")
finally:
    for h in handles:
        if h:
            fd, mm = h
            mm.close()
            os.close(fd)