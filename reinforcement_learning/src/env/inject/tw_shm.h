/**
 * tw_shm.h — Layout de la shared memory entre Python et le hook SDL.
 * /dev/shm/tw_input_<agent_id>
 */
#pragma once
#include <stdint.h>

typedef struct {
    int32_t  direction;      /* -1=gauche, 0=neutre, 1=droite */
    int32_t  jump;           /* 0 ou 1 */
    int32_t  fire;           /* 0 ou 1 */
    int32_t  hook;           /* 0 ou 1 */
    int32_t  weapon_switch;  /* 0=rien, 1=scroll_up, 2=scroll_down */
    float    aim_x;          /* -1..1 */
    float    aim_y;          /* -1..1 */
    int32_t  win_w;
    int32_t  win_h;
    uint64_t seq;
} __attribute__((packed)) TWInputShm;