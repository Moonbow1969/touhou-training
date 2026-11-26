/* Read-only observation of the SHA-pinned official TH10 0.02a trial. */
#include <inttypes.h>
#include <stdio.h>

#include "th10-trial-state.h"

static uint64_t unix_ms(void) {
    FILETIME utc;
    ULARGE_INTEGER value;
    GetSystemTimeAsFileTime(&utc);
    value.LowPart = utc.dwLowDateTime;
    value.HighPart = utc.dwHighDateTime;
    return value.QuadPart / 10000 - UINT64_C(11644473600000);
}

static unsigned held(void) {
    return ((GetAsyncKeyState(VK_LEFT) & 0x8000) ? 1u : 0u)
         | ((GetAsyncKeyState(VK_RIGHT) & 0x8000) ? 2u : 0u)
         | ((GetAsyncKeyState(VK_UP) & 0x8000) ? 4u : 0u)
         | ((GetAsyncKeyState(VK_DOWN) & 0x8000) ? 8u : 0u)
         | ((GetAsyncKeyState(VK_SHIFT) & 0x8000) ? 16u : 0u)
         | ((GetAsyncKeyState('Z') & 0x8000) ? 32u : 0u)
         | ((GetAsyncKeyState('X') & 0x8000) ? 64u : 0u);
}

int main(int argc, char **argv) {
    const int check_only = argc == 2 && !strcmp(argv[1], "--check");
    DWORD pid = 0;
    HANDLE game = th10_process(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, &pid);
    uint16_t dos = 0;
    if (!game || !th10_read(game, TH10_IMAGE, &dos, sizeof(dos)) || dos != 0x5a4d)
        return 1;
    unsigned accepted = 0;
    const uint64_t stop = GetTickCount64() + 14500;
    puts("utc_ms,stage,difficulty,character,shot,lives,x,y,score,held_mask");
    while (GetTickCount64() < stop) {
        struct Th10State state;
        if (th10_read_state(game, &state)) {
            printf("%" PRIu64 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRId32
                   ",%.3f,%.3f,%" PRIu32 ",%u\n", unix_ms(), state.stage, state.difficulty,
                   state.character, state.shot, state.lives, state.xy[0], state.xy[1], state.score, held());
            accepted++;
        }
        if (check_only)
            break;
        Sleep(20);
    }
    CloseHandle(game);
    return accepted >= (check_only ? 1u : 20u) ? 0 : 2;
}
