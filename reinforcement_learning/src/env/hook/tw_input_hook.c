/**
 * tw_input_hook.c
 *
 * Hooke SDL2 pour injecter les inputs dans Teeworlds.
 *
 * Après analyse :
 *   - Fire / weapon switch marchent via SDL_PollEvent (one-shot events)
 *   - Move / jump NE marchent PAS via SDL_GetKeyboardState
 *     → TW lit ses bindings via KEYDOWN/KEYUP events dans SDL_PollEvent
 *   - Aim NE marche PAS via SDL_GetMouseState ni SDL_MOUSEMOTION
 *     → TW utilise SDL_GetRelativeMouseState (mode souris relative)
 *
 * Solution :
 *   - Move / jump → KEYDOWN/KEYUP dans la file SDL_PollEvent
 *   - Fire         → MOUSEBUTTONDOWN/UP one-shot
 *   - Hook         → MOUSEBUTTONDOWN/UP sur edge
 *   - Weapon       → MOUSEWHEEL one-shot
 *   - Aim          → hook SDL_GetRelativeMouseState
 *
 * Compilation :
 *   gcc -shared -fPIC -O2 -o tw_input_hook.so tw_input_hook.c \
 *       -ldl -lrt $(sdl2-config --cflags --libs)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include "tw_shm.h"

/* ── Shared memory ────────────────────────────────────────────────── */
static TWInputShm *g_shm     = NULL;
static uint64_t    g_prev_seq = 0;

/* ── État précédent pour edge detection ──────────────────────────── */
static int32_t g_prev_dir  = 0;
static int32_t g_prev_jump = 0;
static int32_t g_prev_fire = 0;
static int32_t g_prev_hook = 0;
static int32_t g_prev_wpn  = 0;

/* ── Aim relatif accumulé ────────────────────────────────────────── */
/* TW lit SDL_GetRelativeMouseState frame par frame.
   On calcule le delta depuis la position cible précédente. */
static float g_prev_aim_x = 0.0f;
static float g_prev_aim_y = 0.0f;

/* ── File d'événements ───────────────────────────────────────────── */
#define Q_SIZE 64
static SDL_Event g_queue[Q_SIZE];
static int g_q_head = 0, g_q_tail = 0;

static void q_push(const SDL_Event *e) {
    int next = (g_q_tail + 1) % Q_SIZE;
    if (next == g_q_head) return;
    g_queue[g_q_tail] = *e;
    g_q_tail = next;
}
static int q_pop(SDL_Event *e) {
    if (g_q_head == g_q_tail) return 0;
    *e = g_queue[g_q_head];
    g_q_head = (g_q_head + 1) % Q_SIZE;
    return 1;
}

/* ── Helpers ─────────────────────────────────────────────────────── */
static void push_keydown(SDL_Scancode sc, SDL_Keycode sym) {
    SDL_Event e = {0};
    e.type               = SDL_KEYDOWN;
    e.key.state          = SDL_PRESSED;
    e.key.repeat         = 0;
    e.key.keysym.scancode = sc;
    e.key.keysym.sym     = sym;
    q_push(&e);
}
static void push_keyup(SDL_Scancode sc, SDL_Keycode sym) {
    SDL_Event e = {0};
    e.type               = SDL_KEYUP;
    e.key.state          = SDL_RELEASED;
    e.key.repeat         = 0;
    e.key.keysym.scancode = sc;
    e.key.keysym.sym     = sym;
    q_push(&e);
}
static void push_mbtn(Uint8 btn, int down) {
    SDL_Event e = {0};
    e.type          = down ? SDL_MOUSEBUTTONDOWN : SDL_MOUSEBUTTONUP;
    e.button.button = btn;
    e.button.state  = down ? SDL_PRESSED : SDL_RELEASED;
    e.button.clicks = 1;
    q_push(&e);
}
static void push_wheel(int y) {
    SDL_Event e = {0};
    e.type    = SDL_MOUSEWHEEL;
    e.wheel.y = y;
    q_push(&e);
}

/* ── Sync depuis SHM ─────────────────────────────────────────────── */
static void sync_shm(void) {
    if (!g_shm || g_shm->seq == g_prev_seq) return;

    TWInputShm s;
    memcpy(&s, g_shm, sizeof(s));

    fprintf(stderr, "[hook] seq=%lu dir=%d jump=%d fire=%d hook=%d aim=(%.2f,%.2f)\n",
        s.seq, s.direction, s.jump, s.fire, s.hook, s.aim_x, s.aim_y);

    /* ── Direction (A / D) ─────────────────────────────────────────
       KEYUP de l'ancienne direction, KEYDOWN de la nouvelle */
    if (s.direction != g_prev_dir) {
        fprintf(stderr, "[hook] EVENT: dir %d→%d\n", g_prev_dir, s.direction);
        if (g_prev_dir == -1) push_keyup(SDL_SCANCODE_A, SDLK_a);
        if (g_prev_dir ==  1) push_keyup(SDL_SCANCODE_D, SDLK_d);
        if (s.direction == -1) push_keydown(SDL_SCANCODE_A, SDLK_a);
        if (s.direction ==  1) push_keydown(SDL_SCANCODE_D, SDLK_d);
        g_prev_dir = s.direction;
    }

    /* ── Jump (space) ──────────────────────────────────────────────
       À chaque step où jump=1 : KEYUP + KEYDOWN pour recréer un edge.
       TW vérifie m_Jumped&1 — il faut que -jump (KEYUP) passe d'abord
       pour reset le bit, puis +jump (KEYDOWN) pour déclencher le saut. */
    if (s.jump) {
        fprintf(stderr, "[hook] EVENT: KEYUP+KEYDOWN(space)\n");
        if (g_prev_jump) {
            /* Déjà à 1 : créer un edge pour permettre un nouveau saut */
            push_keyup(SDL_SCANCODE_SPACE, SDLK_SPACE);
        }
        push_keydown(SDL_SCANCODE_SPACE, SDLK_SPACE);
    } else if (g_prev_jump) {
        push_keyup(SDL_SCANCODE_SPACE, SDLK_SPACE);
    }
    g_prev_jump = s.jump;

    /* ── Fire (bouton gauche) — tap one-shot ───────────────────────*/
    if (s.fire && !g_prev_fire) {
        push_mbtn(SDL_BUTTON_LEFT, 1);
        push_mbtn(SDL_BUTTON_LEFT, 0);
    }
    g_prev_fire = s.fire;

    /* ── Hook (bouton droit) — hold ────────────────────────────────*/
    if (s.hook != g_prev_hook) {
        push_mbtn(SDL_BUTTON_RIGHT, s.hook ? 1 : 0);
        g_prev_hook = s.hook;
    }

    /* ── Weapon switch — molette ───────────────────────────────────*/
    if (s.weapon_switch && s.weapon_switch != g_prev_wpn) {
        push_wheel(s.weapon_switch == 1 ? 1 : -1);
    }
    g_prev_wpn = s.weapon_switch;

    /* ── Aim : stocker la cible pour SDL_GetRelativeMouseState ─────*/
    g_prev_aim_x = s.aim_x;
    g_prev_aim_y = s.aim_y;

    g_prev_seq = s.seq;
}

/* ── Constructeur ────────────────────────────────────────────────── */
__attribute__((constructor))
static void tw_hook_init(void) {
    const char *aid = getenv("TW_AGENT_ID");
    if (!aid) aid = "0";

    char path[64];
    snprintf(path, sizeof(path), "/dev/shm/tw_input_%s", aid);

    int fd = open(path, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("[tw_hook] open"); return; }
    if (ftruncate(fd, sizeof(TWInputShm)) < 0) { perror("[tw_hook] ftruncate"); return; }

    g_shm = mmap(NULL, sizeof(TWInputShm),
                 PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (g_shm == MAP_FAILED) { perror("[tw_hook] mmap"); g_shm = NULL; return; }

    close(fd);
    memset(g_shm, 0, sizeof(TWInputShm));
    fprintf(stderr, "[tw_hook] OK — shm=%s\n", path);
}

/* ── Hook SDL_PollEvent ──────────────────────────────────────────── */
int SDL_PollEvent(SDL_Event *event) {
    static int (*real)(SDL_Event *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "SDL_PollEvent");

    sync_shm();

    if (q_pop(event)) return 1;
    return real(event);
}

/* ── Hook SDL_GetRelativeMouseState ─────────────────────────────── */
/*
 * TW utilise le mode souris relative pour la visée.
 * À chaque frame, il lit le déplacement depuis le dernier appel.
 * On retourne un delta calculé depuis notre aim_x/aim_y cible.
 */
Uint32 SDL_GetRelativeMouseState(int *x, int *y) {
    static Uint32 (*real)(int *, int *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "SDL_GetRelativeMouseState");

    /* Vider l'état réel (delta physique qu'on ignore) */
    real(x, y);

    Uint32 state = 0;
    if (g_shm) {
        int w = g_shm->win_w > 0 ? g_shm->win_w : 800;
        int h = g_shm->win_h > 0 ? g_shm->win_h : 600;

        /* Position absolue cible en pixels */
        int tx = (int)(w / 2 + g_shm->aim_x * (w / 2));
        int ty = (int)(h / 2 + g_shm->aim_y * (h / 2));

        if (x) *x = tx;
        if (y) *y = ty;

        if (g_shm->hook)
            state |= SDL_BUTTON(SDL_BUTTON_RIGHT);
    }
    return state;
}