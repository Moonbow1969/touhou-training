/* Read the official 0.02a game and recorder settings. No input or write access. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int read_i32(HANDLE game, uintptr_t address, int32_t *value) {
    SIZE_T count;
    return ReadProcessMemory(game, (const void *)address, value, sizeof(*value), &count)
           && count == sizeof(*value);
}

int main(int argc, char **argv) {
    if (argc != 1 && argc != 3) return 2;
    PROCESSENTRY32 process = {0};
    process.dwSize = sizeof process;
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 2;
    DWORD pid = 0;
    if (Process32First(snapshot, &process)) {
        do {
            if (!_stricmp(process.szExeFile, "th12.exe")) {
                if (pid) return 3;
                pid = process.th32ProcessID;
            }
        } while (Process32Next(snapshot, &process));
    }
    CloseHandle(snapshot);
    HANDLE game = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!game) return 4;
    FILE *output = argc == 3 ? fopen(argv[1], "wb") : stdout;
    if (!output) return 5;
    fprintf(output,"ms\trank\trecorded_rank\tstage\tmode\tcharacter\tshot\tframe\tspares\n");
    DWORD start = GetTickCount();
    for (;;) {
        int32_t rank=-1000, replay_rank=-1000, manager=0, header=0;
        int32_t stage=-1000, mode=-1000, character=-1000, shot=-1000, frame=-1000, spares=-1000;
        read_i32(game, 0x4aea30, &rank);
        read_i32(game, 0x4aea38, &stage);
        read_i32(game, 0x4aea18, &character);
        read_i32(game, 0x4aea1c, &shot);
        read_i32(game, 0x4aea20, &spares);
        if (read_i32(game, 0x4b21d4, &manager) && manager>65536) {
            read_i32(game, manager+0x10, &mode);
            read_i32(game, manager+0x100, &frame);
            if (read_i32(game, manager+0x1c, &header) && header>65536) {
                read_i32(game, header+0x64, &replay_rank);
            }
        }
        DWORD elapsed = GetTickCount()-start;
        fprintf(output,"%lu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                (unsigned long)elapsed,rank,replay_rank,stage,mode,character,shot,frame,spares);
        fflush(output);
        if (argc == 1 || GetFileAttributesA(argv[2]) != INVALID_FILE_ATTRIBUTES) break;
        if (elapsed >= 3600000) return 6;
        Sleep(80);
    }
    if (argc == 3) fclose(output);
    CloseHandle(game);
    return 0;
}
