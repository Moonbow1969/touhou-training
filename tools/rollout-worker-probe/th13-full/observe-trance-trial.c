/* Exact TH13 trial read-only control qualification. Every distinct original tick is logged.
 * Gauge/state are hypotheses until independently verified at a real activation. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int read_word(HANDLE p, uintptr_t address, int32_t *value) {
    SIZE_T count=0;
    return ReadProcessMemory(p,(const void*)address,value,sizeof(*value),&count)&&count==sizeof(*value);
}
int main(void) {
    PROCESSENTRY32 pentry={0};pentry.dwSize=sizeof(pentry);
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    DWORD pid=0;
    if(Process32First(snapshot,&pentry)) do {
        if(!_stricmp(pentry.szExeFile,"th13.exe")) {if(pid)return 2;pid=pentry.th32ProcessID;}
    } while(Process32Next(snapshot,&pentry));
    CloseHandle(snapshot);
    if(!pid)return 3;
    MODULEENTRY32 module={0};module.dwSize=sizeof(module);
    snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    if(!Module32First(snapshot,&module))return 4;
    uintptr_t base=(uintptr_t)module.modBaseAddr;CloseHandle(snapshot);
    HANDLE game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if(!game)return 5;
    FILE *out=fopen("Z:/capture/trance-observations.csv","w");if(!out)return 6;
    fprintf(out,"native_tick,stage,field_b90b0,spares,bombs,power,gauge_candidate,active_candidate,native_input,field_b9048,capture_tick_ms\n");
    const uint32_t at[]={0xb90b4,0xb90a8,0xb90b0,0xb9080,0xb908c,0xb9074,0xb9094,0xb9098,0xdf430,0xb9048};
    int32_t prior_tick=-1,prior_stage=-1;
    for(int it=0;it<540000;it++) {
        int32_t a[10],stamp;int ok=1;
        for(int j=0;j<10;j++)if(!read_word(game,base+at[j],&a[j]))ok=0;
        if(!ok||!read_word(game,base+at[0],&stamp))break;
        if((stamp==a[0])&&(a[0]!=prior_tick||a[1]!=prior_stage)){
            for(int j=0;j<10;j++)fprintf(out,"%d,",j==8?(a[j]&65535):a[j]);
            fprintf(out,"%lu\n",(unsigned long)GetTickCount());fflush(out);prior_tick=a[0];prior_stage=a[1];
        }
        if(it%200==0&&GetFileAttributesA("Z:/capture/trance-stop")!=INVALID_FILE_ATTRIBUTES)break;
        Sleep(2);
    }
    fclose(out);CloseHandle(game);return 0;
}
