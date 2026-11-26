#ifndef TH10_TRIAL_STATE_H
#define TH10_TRIAL_STATE_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <tlhelp32.h>

/* Original TH10 0.02a, executable SHA-256 is checked before this reader starts. */
enum {
    TH10_IMAGE = 0x400000,
    TH10_SCORE = 0x472b0c,
    TH10_TITLE_FIELDS = 0x472b30,
    TH10_STAGE = 0x472b44,
    TH10_PLAYER = 0x475594,
    TH10_PLAYER_XY = 0x3c0
};

struct Th10State {
    uint32_t stage, score, character, shot, difficulty;
    int32_t lives;
    float xy[2];
};

static HANDLE th10_process(DWORD access, DWORD *pid) {
    HANDLE all = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (all == INVALID_HANDLE_VALUE)
        return NULL;
    PROCESSENTRY32 row = {0};
    row.dwSize = sizeof(row);
    unsigned matches = 0;
    if (Process32First(all, &row)) {
        do {
            if (!_stricmp(row.szExeFile, "th10tr.exe")) {
                *pid = row.th32ProcessID;
                matches++;
            }
        } while (Process32Next(all, &row));
    }
    CloseHandle(all);
    return matches == 1 ? OpenProcess(access, FALSE, *pid) : NULL;
}

static int th10_read(HANDLE game, uint32_t address, void *value, SIZE_T size) {
    SIZE_T actual;
    return ReadProcessMemory(game, (void *)(uintptr_t)address, value, size, &actual) && actual == size;
}

static int th10_read_state(HANDLE game, struct Th10State *state) {
    uint32_t player;
    uint32_t options[4];
    if (!th10_read(game, TH10_STAGE, &state->stage, sizeof(state->stage)) || state->stage < 1 || state->stage > 3
            || !th10_read(game, TH10_PLAYER, &player, sizeof(player)) || player <= 0x10000
            || !th10_read(game, player + TH10_PLAYER_XY, state->xy, sizeof(state->xy))
            || !th10_read(game, TH10_SCORE, &state->score, sizeof(state->score))
            || !th10_read(game, TH10_TITLE_FIELDS, options, sizeof(options)))
        return 0;
    state->character = options[0];
    state->shot = options[1];
    state->lives = (int32_t)options[2];
    state->difficulty = options[3];
    return state->score <= 999999999 && state->difficulty <= 4 && isfinite(state->xy[0])
        && fabsf(state->xy[0]) <= 200.f && state->xy[1] >= 0.f && state->xy[1] <= 450.f;
}
#endif
