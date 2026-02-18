/**
 * tw_input_hook.c
 *
 * Hooke trois fonctions SDL2 :
 *   - SDL_PollEvent        → événements one-shot (fire, wheel, aim)
 *   - SDL_GetKeyboardState → état continu clavier (move, jump)
 *   - SDL_GetMouseState    → état continu souris  (hook)
 *
 * Compilation :
 *   gcc -shared -fPIC -O2 -o tw_input_hook.so tw_input_hook.c \
 *       -ldl -lrt $(sdl2-config --cflags --libs)
 *
 * Lancement :
 *   TW_AGENT_ID=0 LD_PRELOAD=/path/to/tw_input_hook.so teeworlds ...
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

/* ── Edge detection pour les one-shot ───────────────────────────── */
static int32_t g_prev_fire = 0;
static int32_t g_prev_hook = 0;
static int32_t g_prev_wpn  = 0;

/* ── Faux état clavier retourné par SDL_GetKeyboardState ─────────── */
static Uint8 g_fake_keys[512] = {0};

/* ── File d'événements one-shot ──────────────────────────────────── */
#define Q_SIZE 32
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

static void push_mbtn(Uint8 btn, int down) {
    SDL_Event e = {0};
    e.type          = down ? SDL_MOUSEBUTTONDOWN : SDL_MOUSEBUTTONUP;
    e.button.button = btn;
    e.button.state  = down ? SDL_PRESSED : SDL_RELEASED;
    q_push(&e);
}
static void push_wheel(int y) {
    SDL_Event e = {0};
    e.type    = SDL_MOUSEWHEEL;
    e.wheel.y = y;
    q_push(&e);
}

/* ── Mise à jour principale depuis la SHM ────────────────────────── */
static void sync_shm(void) {
    if (!g_shm || g_shm->seq == g_prev_seq) return;

    TWInputShm s;
    memcpy(&s, g_shm, sizeof(s));

    /* Clavier continu — directement dans le faux tableau */
    g_fake_keys[SDL_SCANCODE_A]     = (s.direction == -1) ? 1 : 0;
    g_fake_keys[SDL_SCANCODE_D]     = (s.direction ==  1) ? 1 : 0;
    g_fake_keys[SDL_SCANCODE_SPACE] = s.jump ? 1 : 0;

    /* Fire — tap souris sur front montant */
    if (s.fire && !g_prev_fire) {
        push_mbtn(SDL_BUTTON_LEFT, 1);
        push_mbtn(SDL_BUTTON_LEFT, 0);
    }
    g_prev_fire = s.fire;

    /* Hook — hold souris */
    if (s.hook != g_prev_hook) {
        push_mbtn(SDL_BUTTON_RIGHT, s.hook ? 1 : 0);
        g_prev_hook = s.hook;
    }

    /* Weapon switch — molette sur front montant */
    if (s.weapon_switch && s.weapon_switch != g_prev_wpn) {
        push_wheel(s.weapon_switch == 1 ? 1 : -1);
    }
    g_prev_wpn = s.weapon_switch;

    /* Aim — MOUSEMOTION */
    {
        int w = s.win_w > 0 ? s.win_w : 800;
        int h = s.win_h > 0 ? s.win_h : 600;
        SDL_Event e = {0};
        e.type     = SDL_MOUSEMOTION;
        e.motion.x = (int)(w / 2 + s.aim_x * (w / 3));
        e.motion.y = (int)(h / 2 + s.aim_y * (h / 3));
        q_push(&e);
    }

    g_prev_seq = s.seq;

    fprintf(stderr, "[hook] seq=%lu dir=%d jump=%d space_key=%d\n",
        s.seq, s.direction,
        s.jump, g_fake_keys[SDL_SCANCODE_SPACE]);
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

/* ── Hook SDL_GetKeyboardState ───────────────────────────────────── */
/*
 * TW lit l'état continu du clavier via cette fonction.
 * On retourne notre faux tableau au lieu de l'état physique.
 */
const Uint8 *SDL_GetKeyboardState(int *numkeys) {
    if (numkeys) *numkeys = 512;
    sync_shm();
    return g_fake_keys;
}

/* ── Hook SDL_GetMouseState ──────────────────────────────────────── */
/*
 * TW lit l'état continu des boutons souris via cette fonction.
 * On injecte le bouton droit (hook) et la position (aim).
 */
Uint32 SDL_GetMouseState(int *x, int *y) {
    static Uint32 (*real)(int *, int *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "SDL_GetMouseState");

    /* Position réelle (on l'ignore, on override ci-dessous) */
    real(x, y);

    Uint32 state = 0;
    if (g_shm) {
        if (g_shm->hook)
            state |= SDL_BUTTON(SDL_BUTTON_RIGHT);

        if (x && y) {
            int w = g_shm->win_w > 0 ? g_shm->win_w : 800;
            int h = g_shm->win_h > 0 ? g_shm->win_h : 600;
            *x = (int)(w / 2 + g_shm->aim_x * (w / 3));
            *y = (int)(h / 2 + g_shm->aim_y * (h / 3));
        }
    }
    return state;
}