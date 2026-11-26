/* Frame-independent 85 ms control probe for the official TH10 0.02a trial. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "th10-trial-state.h"

enum {
    BULLET_MANAGER = 0x475450,
    BULLET_POOL = 0x60,
    BULLET_STRIDE = 0x7bc,
    BULLET_SLOTS = 2001,
    BULLET_XY = 0x3b4,
    BULLET_VXY = 0x3c0,
    BULLET_ACTIVE = 0x446,
    CAPACITY = 96,
    HOLD_MS = 85
};

struct Bullet {
    float relative[2], velocity[2], distance;
};
static char pool[BULLET_SLOTS * BULLET_STRIDE];
static struct Bullet objects[BULLET_SLOTS];

static int compare_distance(const void *a, const void *b) {
    float left = ((const struct Bullet *)a)->distance;
    float right = ((const struct Bullet *)b)->distance;
    return (left > right) - (left < right);
}

static int projectiles(HANDLE game, const float xy[2]) {
    uint32_t manager;
    if (!th10_read(game, BULLET_MANAGER, &manager, sizeof(manager)) || !manager
            || !th10_read(game, manager + BULLET_POOL, pool, sizeof(pool)))
        return -1;
    int count = 0;
    for (unsigned i = 0; i < BULLET_SLOTS; i++) {
        const char *slot = pool + BULLET_STRIDE * i;
        uint16_t active;
        memcpy(&active, slot + BULLET_ACTIVE, sizeof(active));
        if (active != 1)
            continue;
        struct Bullet item;
        memcpy(item.relative, slot + BULLET_XY, sizeof(item.relative));
        memcpy(item.velocity, slot + BULLET_VXY, sizeof(item.velocity));
        for (unsigned axis = 0; axis < 2; axis++) {
            item.relative[axis] -= xy[axis];
            if (!isfinite(item.relative[axis]) || !isfinite(item.velocity[axis])
                    || fabsf(item.relative[axis]) > 1800.f || fabsf(item.velocity[axis]) > 100.f)
                return -1;
        }
        item.distance = item.relative[0] * item.relative[0] + item.relative[1] * item.relative[1];
        objects[count++] = item;
    }
    qsort(objects, count, sizeof(objects[0]), compare_distance);
    return count;
}

static int emit(HANDLE game, unsigned held, uint64_t elapsed) {
    struct Th10State state;
    if (!th10_read_state(game, &state) || state.difficulty != 1 || state.character != 0 || state.shot != 0) {
        fputs("TH10 readout unavailable at the native Normal Reimu A state.\n", stderr);
        return 0;
    }
    int count = projectiles(game, state.xy);
    if (count < 0) {
        fputs("TH10 projectile pool readout failed validation.\n", stderr);
        return 0;
    }
    printf("{\"stage\":%" PRIu32 ",\"lives\":%" PRId32 ",\"score\":%" PRIu32
           ",\"position\":[%.4f,%.4f],\"time_ms\":%" PRIu64 ",\"held\":%u,\"bullet_count\":%d,\"bullets\":[",
           state.stage, state.lives, state.score, state.xy[0], state.xy[1], elapsed, held, count);
    for (int i = 0; i < count && i < CAPACITY; i++) {
        const struct Bullet *b = objects + i;
        printf("%s[%.4f,%.4f,%.4f,%.4f]", i ? "," : "", b->relative[0], b->relative[1],
               b->velocity[0], b->velocity[1]);
    }
    puts("]}");
    fflush(stdout);
    return 1;
}

int main(void) {
    DWORD pid = 0;
    HANDLE game = th10_process(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, &pid);
    if (!game || !DebugActiveProcess(pid)) {
        fprintf(stderr, "Could not attach to the approved private TH10 game (error %lu).\n",
                (unsigned long)GetLastError());
        return 1;
    }
    DebugSetProcessKillOnExit(FALSE);
    int stopped = 1;
    int ok = emit(game, 0, 0);
    char line[64];
    while (ok && fgets(line, sizeof(line), stdin)) {
        if (line[0] == 'Q')
            break;
        unsigned next;
        ok = sscanf(line, "S %u", &next) == 1 && next >= 32 && next <= 63;
        if (!ok || !DebugActiveProcessStop(pid)) {
            fprintf(stderr, "TH10 could not resume the requested action (error %lu).\n", (unsigned long)GetLastError());
            ok = 0;
            break;
        }
        stopped = 0;
        const uint64_t start = GetTickCount64();
        Sleep(HOLD_MS);
        stopped = DebugActiveProcess(pid);
        if (!stopped)
            fprintf(stderr, "TH10 could not pause at the action boundary (error %lu).\n", (unsigned long)GetLastError());
        ok = stopped && emit(game, next, GetTickCount64() - start);
    }
    if (stopped)
        DebugActiveProcessStop(pid);
    CloseHandle(game);
    return ok ? 0 : 2;
}
