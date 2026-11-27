#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#ifdef TRIAL_CAMERA
#include "th14-camera.h"
#endif

#ifdef TRIAL_CONTACT_TELEMETRY
/* Verified against the SHA-pinned trial's player-state getter and writes. */
enum { PLAYER_CONTACT_STATE = 0x684, GAME_BOMBS = 0x4f1f90 };
#endif

/* Addresses for the unmodified TH14 0.01b trial at image base 0x400000. */
enum {
    CLOCK_POINTER = 0x4d7c40, THREAD_POINTER = 0x4d7c78,
    PLAYER_POINTER = 0x4d7d9c, BULLET_POINTER = 0x4d7c50,
    GAME_STATE = 0x4f41e8, GAME_STAGE = 0x4f1fc4, GAME_LIVES = 0x4f1f84,
    GAME_DIFFICULTY = 0x4f1f54, GAME_CHARACTER = 0x4f1f48, GAME_SHOT = 0x4f1f4c,
    CLOCK_FIELD = 0x191e0, STAGE_CLOCK = 0x14, PLAYER_POSITION = 0x5e0,
    BULLET_LIST = 0x7c, BULLET_FIELDS = 0xbc0, BULLET_FIELDS_SIZE = 0x50,
    MAX_ENTITIES = 96, MAX_LIST_LENGTH = 8192
};

struct Target {
    HWND window;
    DWORD pid;
    HANDLE process;
    int frozen;
    unsigned int held;
};

struct State {
    uint32_t clock;
    uint32_t frame;
    uint32_t mode;
    uint32_t stage;
    int32_t lives;
    float position[2];
#ifdef TRIAL_CONTACT_TELEMETRY
    uint32_t player_state;
    int32_t bombs;
#endif
};

struct Entity {
    float x, y, vx, vy, radius;
    uint16_t active;
    float distance;
};

static struct Entity entities[MAX_LIST_LENGTH];
static const WORD keys[] = {VK_LEFT, VK_RIGHT, VK_UP, VK_DOWN, VK_SHIFT, 'Z', 'X'};

static BOOL CALLBACK find_trial(HWND window, LPARAM value) {
    char title[128] = {0};
    GetWindowTextA(window, title, sizeof(title));
    if (strstr(title, "Double Dealing Character. ver 0.01b") && IsWindowVisible(window)) {
        *(HWND *)value = window;
        return FALSE;
    }
    return TRUE;
}

static int read_memory(const struct Target *game, uint32_t address, void *value, SIZE_T size) {
    SIZE_T transferred = 0;
    return ReadProcessMemory(game->process, (void *)(uintptr_t)address, value, size, &transferred)
           && transferred == size;
}

static int read_field(const struct Target *game, uint32_t pointer, uint32_t offset, void *out, SIZE_T size) {
    uint32_t address;
    return read_memory(game, pointer, &address, sizeof(address)) && address
           && read_memory(game, address + offset, out, size);
}

static int snapshot(const struct Target *game, struct State *state) {
    return read_field(game, CLOCK_POINTER, CLOCK_FIELD, &state->clock, sizeof(state->clock))
           && read_field(game, THREAD_POINTER, STAGE_CLOCK, &state->frame, sizeof(state->frame))
           && read_field(game, PLAYER_POINTER, PLAYER_POSITION, state->position, sizeof(state->position))
           && read_memory(game, GAME_STATE, &state->mode, sizeof(state->mode))
           && read_memory(game, GAME_STAGE, &state->stage, sizeof(state->stage))
           && read_memory(game, GAME_LIVES, &state->lives, sizeof(state->lives))
#ifdef TRIAL_CONTACT_TELEMETRY
           && read_field(game, PLAYER_POINTER, PLAYER_CONTACT_STATE, &state->player_state, sizeof(state->player_state))
           && read_memory(game, GAME_BOMBS, &state->bombs, sizeof(state->bombs))
#endif
           && isfinite(state->position[0]) && isfinite(state->position[1])
           && fabsf(state->position[0]) <= 250.0f && fabsf(state->position[1]) <= 520.0f;
}

static int compare_distance(const void *left, const void *right) {
    float a = ((const struct Entity *)left)->distance;
    float b = ((const struct Entity *)right)->distance;
    return (a > b) - (a < b);
}

static int observe_entities(const struct Target *game, const struct State *state) {
    uint32_t manager;
    if (!read_memory(game, BULLET_POINTER, &manager, sizeof(manager)))
        return -1;
    if (!manager)
        return 0;
    uint32_t address = manager + BULLET_LIST;
    int count = 0;
    for (int i = 0; i < MAX_LIST_LENGTH; i++) {
        uint32_t node[2];
        if (!read_memory(game, address, node, sizeof(node)))
            return -1;
        if (node[0]) {
            unsigned char data[BULLET_FIELDS_SIZE];
            if (!read_memory(game, node[0] + BULLET_FIELDS, data, sizeof(data)))
                return -1;
            struct Entity *bullet = &entities[count++];
            memcpy(&bullet->x, data, 4);
            memcpy(&bullet->y, data + 4, 4);
            memcpy(&bullet->vx, data + 0xc, 4);
            memcpy(&bullet->vy, data + 0x10, 4);
            memcpy(&bullet->radius, data + 0x20, 4);
            memcpy(&bullet->active, data + 0x4e, 2);
            if (!isfinite(bullet->x) || !isfinite(bullet->y) || !isfinite(bullet->vx)
                || !isfinite(bullet->vy) || !isfinite(bullet->radius))
                return -1;
            bullet->x -= state->position[0];
            bullet->y -= state->position[1];
            bullet->distance = bullet->x * bullet->x + bullet->y * bullet->y;
        }
        address = node[1];
        if (!address) {
            qsort(entities, count, sizeof(entities[0]), compare_distance);
            return count;
        }
    }
    return -1;
}

static int emit_state(const struct Target *game, const struct State *state, uint32_t delta) {
#ifdef TRIAL_CAMERA
    int count=0;
#else
    int count = observe_entities(game, state);
    if (count < 0)
        return 0;
#endif
    printf("{\"ticks\":%lu,\"frame\":%lu,\"delta\":%lu,\"mode\":%lu,\"stage\":%lu,"
           "\"lives\":%ld,\"player\":[%.4f,%.4f],\"bullet_count\":%d,",
           (unsigned long)state->clock, (unsigned long)state->frame, (unsigned long)delta,
           (unsigned long)state->mode, (unsigned long)state->stage, (long)state->lives,
           state->position[0], state->position[1], count);
#ifdef TRIAL_CONTACT_TELEMETRY
    printf("\"contact_schema\":1,\"player_state\":%lu,\"bombs\":%ld,\"held\":%u,",
           (unsigned long)state->player_state, (long)state->bombs, game->held);
#endif
#ifdef TRIAL_CAMERA
    if(!emit_camera(game->window))return 0;
#endif
    fputs("\"bullets\":[", stdout);
    for (int i = 0; i < count && i < MAX_ENTITIES; i++) {
        const struct Entity *b = &entities[i];
        printf("%s[%.4f,%.4f,%.4f,%.4f,%.4f,%d]", i ? "," : "",
               b->x, b->y, b->vx, b->vy, b->radius, b->active == 1);
    }
    puts("]}");
    fflush(stdout);
    return 1;
}

static int key_event(WORD key, int down) {
    INPUT input = {0};
    input.type = INPUT_KEYBOARD;
    input.ki.wScan = (WORD)MapVirtualKeyA(key, MAPVK_VK_TO_VSC);
    input.ki.dwFlags = KEYEVENTF_SCANCODE | (down ? 0 : KEYEVENTF_KEYUP);
    if (key >= VK_LEFT && key <= VK_DOWN)
        input.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;
    return SendInput(1, &input, sizeof(input)) == 1;
}

static int set_keys(struct Target *game, unsigned int held) {
    for (unsigned int i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        unsigned int bit = 1u << i;
        if ((held ^ game->held) & bit) {
            if (!key_event(keys[i], (held & bit) != 0))
                return 0;
            game->held ^= bit;
        }
    }
    return 1;
}

static int freeze(struct Target *game) {
    if (!DebugActiveProcess(game->pid))
        return 0;
    game->frozen = 1;
    return DebugSetProcessKillOnExit(FALSE);
}

static int thaw(struct Target *game) {
    if (game->frozen && !DebugActiveProcessStop(game->pid))
        return 0;
    game->frozen = 0;
    return 1;
}

static int ready(const struct Target *game, struct State *state) {
    return snapshot(game, state) && state->mode == 2 && state->stage == 1
           && state->lives >= 0;
}

static int wait_ticks(const struct Target *game, uint32_t before, uint32_t ticks) {
    ULONGLONG deadline = GetTickCount64() + 4000;
    while (GetTickCount64() < deadline) {
        uint32_t now;
        if (!read_field(game, CLOCK_POINTER, CLOCK_FIELD, &now, sizeof(now)))
            return 0;
        if (now - before >= ticks)
            return 1;
        Sleep(1);
    }
    return 0;
}

static int tap(const struct Target *game, WORD key) {
    uint32_t clock;
    if (!read_field(game, CLOCK_POINTER, CLOCK_FIELD, &clock, sizeof(clock)) || !key_event(key, 1))
        return 0;
    int advanced = wait_ticks(game, clock, 5);
    return key_event(key, 0) && advanced;
}

static int start_stage(struct Target *game, struct State *state) {
    ULONGLONG deadline = GetTickCount64() + 55000;
    uint32_t next_selection = 90;
    while (GetTickCount64() < deadline) {
        if (ready(game, state)) {
            uint32_t rank, character, shot;
            return read_memory(game, GAME_DIFFICULTY, &rank, sizeof(rank)) && rank == 1
                   && read_memory(game, GAME_CHARACTER, &character, sizeof(character)) && character == 0
                   && read_memory(game, GAME_SHOT, &shot, sizeof(shot)) && shot == 0;
        }
        uint32_t clock;
        if (GetForegroundWindow() == game->window
            && read_field(game, CLOCK_POINTER, CLOCK_FIELD, &clock, sizeof(clock))
            && clock >= next_selection) {
            if (!tap(game, 'Z'))
                return 0;
            next_selection = clock + 74;
        }
        Sleep(20);
    }
    return 0;
}

static int reset_stage(struct Target *game, struct State *state) {
    uint32_t previous_frame = state->frame;
    if (GetForegroundWindow() != game->window || !set_keys(game, 0) || !thaw(game))
        return 0;
    if (!tap(game, VK_ESCAPE))
        return 0;
    uint32_t pause_clock;
    if (!read_field(game, CLOCK_POINTER, CLOCK_FIELD, &pause_clock, sizeof(pause_clock))
        || !wait_ticks(game, pause_clock, 36) || !snapshot(game, state) || state->mode != 0) {
        fputs("The native pause menu did not become ready for Retry.\n", stderr);
        return 0;
    }
    if (!tap(game, 'R'))
        return 0;
    ULONGLONG deadline = GetTickCount64() + 7000;
    while (GetTickCount64() < deadline) {
        if (ready(game, state) && state->frame < 90 && state->frame < previous_frame)
            return freeze(game) && snapshot(game, state) && emit_state(game, state, 0);
        Sleep(10);
    }
    fputs("Retry did not reset the native stage clock.\n", stderr);
    return 0;
}

static int step(struct Target *game, struct State *state, unsigned int held) {
    uint32_t before = state->clock;
    if (GetForegroundWindow() != game->window || !thaw(game) || !set_keys(game, held))
        return 0;
    return wait_ticks(game, before, 4) && freeze(game) && snapshot(game, state)
           && emit_state(game, state, state->clock - before);
}

int main(void) {
    struct Target game = {0};
    EnumWindows(find_trial, (LPARAM)&game.window);
    if (!game.window || !GetWindowThreadProcessId(game.window, &game.pid))
        return 2;
    game.process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, game.pid);
    if (!game.process)
        return 3;

    struct State state;
    if (!start_stage(&game, &state) || !freeze(&game) || !snapshot(&game, &state)
        || !emit_state(&game, &state, 0)) {
        fputs("TH14 trial did not reach a readable Normal Stage 1 state.\n", stderr);
        thaw(&game);
        CloseHandle(game.process);
        return 4;
    }

    int result = 0;
    char line[80];
    while (fgets(line, sizeof(line), stdin)) {
        unsigned int held;
        if (line[0] == 'Q')
            break;
        if (line[0] == 'R' && reset_stage(&game, &state))
            continue;
        if (sscanf(line, "S %u", &held) == 1 && held <= 127 && step(&game, &state, held))
            continue;
        fputs("TH14 trial request failed.\n", stderr);
        result = 5;
        break;
    }
    set_keys(&game, 0);
    thaw(&game);
    CloseHandle(game.process);
    return result;
}
