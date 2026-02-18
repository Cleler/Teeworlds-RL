/**
 * tw_input_hook.c
 *
 * Injecté via LD_PRELOAD dans le client Teeworlds.
 * Intercepte SDL_PollEvent et injecte des SDL_Event synthétiques
 * à partir de la shared memory écrite par Python.
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

/* ── File d'événements synthétiques ───────────────────────────────── */
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

/* ── Shared memory ────────────────────────────────────────────────── */
static TWInputShm *g_shm = NULL;

/* ── État précédent ──────────────────────────────────────────────── */
static uint64_t g_prev_seq   = 0;
static int32_t  g_prev_dir   = 0;
static int32_t  g_prev_jump  = 0;
static int32_t  g_prev_fire  = 0;
static int32_t  g_prev_hook  = 0;
static int32_t  g_prev_wpn   = 0;

/* ── Helpers ─────────────────────────────────────────────────────── */
static void push_keydown(SDL_Keycode sym) {
    SDL_Event e = {0};
    e.type = SDL_KEYDOWN;
    e.key.state = SDL_PRESSED;
    e.key.keysym.sym = sym;
    e.key.keysym.scancode = SDL_GetScancodeFromKey(sym);
    q_push(&e);
}

static void push_keyup(SDL_Keycode sym) {
    SDL_Event e = {0};
    e.type = SDL_KEYUP;
    e.key.state = SDL_RELEASED;
    e.key.keysym.sym = sym;
    e.key.keysym.scancode = SDL_GetScancodeFromKey(sym);
    q_push(&e);
}

static void push_mbtn(Uint8 btn, int down) {
    SDL_Event e = {0};
    e.type = down ? SDL_MOUSEBUTTONDOWN : SDL_MOUSEBUTTONUP;
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

static void push_motion(int x, int y) {
    SDL_Event e = {0};
    e.type     = SDL_MOUSEMOTION;
    e.motion.x = x;
    e.motion.y = y;
    q_push(&e);
}

/* ── Génération des événements depuis la SHM ─────────────────────── */
static void generate_events(void) {
    TWInputShm s;
    memcpy(&s, g_shm, sizeof(s));

    /* Direction — KEYDOWN/KEYUP sur changement */
    if (s.direction != g_prev_dir) {
        if (g_prev_dir == -1) push_keyup(SDLK_a);
        if (g_prev_dir ==  1) push_keyup(SDLK_d);
        if (s.direction == -1) push_keydown(SDLK_a);
        if (s.direction ==  1) push_keydown(SDLK_d);
        g_prev_dir = s.direction;
    }

    /* Jump — hold : KEYDOWN quand 0→1, KEYUP quand 1→0 */
    if (s.jump != g_prev_jump) {
        if (s.jump) push_keydown(SDLK_SPACE);
        else        push_keyup(SDLK_SPACE);
        g_prev_jump = s.jump;
    }

    /* Fire — tap sur front montant uniquement */
    if (s.fire && !g_prev_fire) {
        push_mbtn(SDL_BUTTON_LEFT, 1);
        push_mbtn(SDL_BUTTON_LEFT, 0);
    }
    g_prev_fire = s.fire;

    /* Hook — hold : MOUSEDOWN quand 0→1, MOUSEUP quand 1→0 */
    if (s.hook != g_prev_hook) {
        push_mbtn(SDL_BUTTON_RIGHT, s.hook ? 1 : 0);
        g_prev_hook = s.hook;
    }

    /* Weapon switch — tap sur front montant */
    if (s.weapon_switch && s.weapon_switch != g_prev_wpn) {
        push_wheel(s.weapon_switch == 1 ? 1 : -1);
    }
    g_prev_wpn = s.weapon_switch;

    /* Aim — MOUSEMOTION à chaque step */
    int w  = s.win_w > 0 ? s.win_w : 800;
    int h  = s.win_h > 0 ? s.win_h : 600;
    int tx = (int)(w / 2 + s.aim_x * (w / 3));
    int ty = (int)(h / 2 + s.aim_y * (h / 3));
    push_motion(tx, ty);

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

/* ── Hook SDL_PollEvent ───────────────────────────────────────────── */
int SDL_PollEvent(SDL_Event *event) {
    static int (*real)(SDL_Event *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "SDL_PollEvent");

    /* Vider d'abord la file synthétique */
    if (q_pop(event)) return 1;

    int ret = real(event);

    /* Si nouvelle séquence disponible, générer les événements */
    if (g_shm && g_shm->seq != g_prev_seq) {
        generate_events();
        if (!ret && q_pop(event)) return 1;
    }

    return ret;
}