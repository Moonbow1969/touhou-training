/* Timestamp each change of native trial clock/input. Observation only, never feeds the actor. */
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int open_game(void) {
    DIR *dir = opendir("/proc");
    if (!dir) return -1;
    struct dirent *entry;
    int fd = -1;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] < '1' || entry->d_name[0] > '9') continue;
        char path[320], name[64];
        snprintf(path, sizeof(path), "/proc/%s/comm", entry->d_name);
        FILE *status = fopen(path, "r");
        if (!status) continue;
        int matched = fgets(name, sizeof(name), status) && strcmp(name, "th07.exe\n") == 0;
        fclose(status);
        if (!matched) continue;
        snprintf(path, sizeof(path), "/proc/%s/mem", entry->d_name);
        fd = open(path, O_RDONLY);
        if (fd >= 0) break;
    }
    closedir(dir);
    return fd;
}

static long long mono_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (long long)now.tv_sec * 1000000000 + now.tv_nsec;
}

int main(void) {
    int fd = open_game();
    if (fd < 0) return 2;
    FILE *out = fopen("/capture/native-input.csv", "wx");
    if (!out) return 3;
    fprintf(out, "lo_ns,hi_ns,tick,word,previous_word\n");
    uint32_t last_tick = UINT32_MAX, last_word = UINT32_MAX;
    unsigned rows = 0;
    while (access("/capture/input-sampler.stop", F_OK) != 0) {
        uint32_t tick, next_tick, word, prev, second_word;
        long long lo = mono_ns();
        if (pread(fd, &tick, 4, 0x11c9ae8) != 4 ||
            pread(fd, &word, 4, 0x11c9d24) != 4 ||
            pread(fd, &prev, 4, 0x11c9d2c) != 4 ||
            pread(fd, &second_word, 4, 0x11c9d24) != 4 ||
            pread(fd, &next_tick, 4, 0x11c9ae8) != 4) return 4;
        long long hi = mono_ns();
        if (tick == next_tick && word == second_word && (tick != last_tick || word != last_word)) {
            if (fprintf(out, "%lld,%lld,%u,%u,%u\n", lo, hi, tick, word & 65535, prev & 65535) < 0) return 5;
            last_tick = tick;
            last_word = word;
            if (++rows % 32 == 0 && fflush(out)) return 6;
        }
        usleep(1500);
    }
    close(fd);
    return fclose(out) || !rows;
}
