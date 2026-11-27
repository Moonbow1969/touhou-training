/* Read-only qualification probes for original TH10 trial 0.02a. */
#include <inttypes.h>
#include <stdio.h>
#include "th10-trial-state.h"

static uint64_t clock_ms(void) {
    FILETIME stamp; ULARGE_INTEGER value;
    GetSystemTimeAsFileTime(&stamp); value.HighPart=stamp.dwHighDateTime; value.LowPart=stamp.dwLowDateTime;
    return value.QuadPart/10000-UINT64_C(11644473600000);
}

int main(void) {
    DWORD pid=0; HANDLE game=th10_process(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,&pid);
    if(!game)return 2;
    /* Retail thprac's STD +0x38 timer and replay-manager pointer, relocated relative
       to this trial's independently identified player global. Values remain candidates. */
    const uint32_t pointers[]={0x475444,0x475448,0x47544c,0x475598};
    const uint32_t offsets[]={0x30,0x34,0x38,0x3c,0x40,0x44,0x48,0x4c,0x50,0x1c0,0x1c4,0x1c8,0x1cc,0x1d0};
    printf("utc_ms,stage,spares,x,y");
    for(size_t p=0;p<sizeof pointers/sizeof pointers[0];p++)
      for(size_t o=0;o<sizeof offsets/sizeof offsets[0];o++)printf(",p%x_o%x",pointers[p],offsets[o]);
    putchar('\n');fflush(stdout);
    uint64_t start=GetTickCount64();
    while(GetTickCount64()-start<15*60*1000 && GetFileAttributesA("Z:\\capture\\clock-stop")==INVALID_FILE_ATTRIBUTES) {
        struct Th10State state={0};
        if(!th10_read_state(game,&state)){Sleep(20);continue;}
        printf("%" PRIu64 ",%u,%d,%.2f,%.2f",clock_ms(),state.stage,state.lives,state.xy[0],state.xy[1]);
        for(size_t p=0;p<sizeof pointers/sizeof pointers[0];p++){
            uint32_t address=0;int32_t values[0x1d4/4];
            int ok=th10_read(game,pointers[p],&address,4) && address>0x10000 &&
                th10_read(game,address,values,sizeof values);
            for(size_t o=0;o<sizeof offsets/sizeof offsets[0];o++){
                if(ok)printf(",%d",values[offsets[o]/4]);else putchar(',');
            }
        }
        putchar('\n');fflush(stdout);Sleep(18);
    }
    CloseHandle(game);return 0;
}
