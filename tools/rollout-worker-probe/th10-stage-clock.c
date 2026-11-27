/* Opt-in original TH10 trial stage clock. It never supplies observations to the actor. */
#include <inttypes.h>
#include <stdio.h>
#include "th10-trial-state.h"

static uint64_t utc_ms(void) {
    FILETIME filetime;
    ULARGE_INTEGER ticks;
    GetSystemTimeAsFileTime(&filetime);
    ticks.LowPart=filetime.dwLowDateTime;
    ticks.HighPart=filetime.dwHighDateTime;
    return ticks.QuadPart/10000-UINT64_C(11644473600000);
}

int main(void) {
    DWORD pid=0;
    HANDLE game=th10_process(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,&pid);
    if(!game)return 2;
    puts("utc_ms,stage,stage_tick,lives,difficulty");
    fflush(stdout);
    const uint64_t start=GetTickCount64();
    int samples=0;
    while(GetTickCount64()-start<45*60*1000) {
        /* One contiguous native read starting at 0x472b38 captures stock, rank,
           current stage and the independent 0x472b4c stage-local tick together. */
        int32_t fields[6];
        if(th10_read(game, TH10_TITLE_FIELDS+8, fields, sizeof(fields))) {
            int lives=fields[0],rank=fields[1],stage=fields[2],tick=fields[5];
            if(rank>=0 && rank<=3 && stage>=1 && stage<=3 && tick>=0) {
                printf("%" PRIu64 ",%d,%d,%d,%d\n",utc_ms(),stage,tick,lives,rank);
                fflush(stdout);
                samples++;
                if(lives<0)break;
            }
        }
        if(GetFileAttributesA("Z:\\capture\\actor-closed")!=INVALID_FILE_ATTRIBUTES)break;
        Sleep(16);
    }
    CloseHandle(game);
    return samples?0:3;
}
