/* Read-only telemetry for the unmodified official TH13 Trial 0.01a executable.
   Build verified by the launcher. Counters were calibrated against native input, HUD inventory, and the pause menu. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static int get(HANDLE process, uintptr_t at, void *dest, size_t length) {
    SIZE_T n = 0;
    return ReadProcessMemory(process, (const void *)at, dest, length, &n) && n == length;
}
static int32_t scalar(HANDLE p, uintptr_t at) {
    int32_t v = -999;
    get(p, at, &v, sizeof(v));
    return v;
}
static DWORD pid_of_game(void) {
    PROCESSENTRY32 proc = {0}; proc.dwSize=sizeof(proc);
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    DWORD pid=0;
    if (Process32First(snapshot,&proc)) do {
        if (!_stricmp(proc.szExeFile,"th13.exe")) {
            if (pid) { pid=0; break; }
            pid=proc.th32ProcessID;
        }
    } while (Process32Next(snapshot,&proc));
    CloseHandle(snapshot);
    return pid;
}
int main(void) {
    DWORD pid=pid_of_game();
    if (!pid) return 1;
    MODULEENTRY32 entry={0};entry.dwSize=sizeof(entry);
    HANDLE modules=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    if (!Module32First(modules,&entry)) return 2;
    uintptr_t base=(uintptr_t)entry.modBaseAddr;
    CloseHandle(modules);
    HANDLE p=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if (!p || scalar(p,base)==-999) return 3;
    FILE *state=fopen("Z:/capture/telemetry.bin","wb+");
    if (!state) return 4;
    for (int id=1;;++id) {
        int32_t r[21]={0};
        int32_t stock[28]={0};
        if (!get(p, base+0xb9048, stock, sizeof(stock))) break;
        r[0]=id; r[1]=scalar(p,base+0xd7464); r[2]=scalar(p,base+0xd9960);
        r[3]=stock[24]; r[4]=stock[14]; r[5]=stock[17]; r[6]=stock[2];
        r[7]=stock[4]; r[8]=stock[3]; r[9]=stock[0]; r[10]=scalar(p,base+0xdf218)&0xffff;
        r[11]=stock[27]; r[12]=-1; r[13]=0; r[14]=0; r[15]=-1;
        r[16]=stock[15]; r[17]=stock[18]; r[18]=stock[11]; r[19]=stock[26];
        r[20]=id;
        rewind(state);
        if (fwrite(r,sizeof(r),1,state)!=1 || fflush(state)) break;
        Sleep(12);
    }
    fclose(state);CloseHandle(p);return 0;
}
