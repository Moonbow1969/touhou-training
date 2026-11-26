/* Rich, read-only targets for matching one original TH10 training camera to replay state. */
#define main th10_step_entry_unused
#include "th10-trial-step.c"
#undef main

static uint64_t utc_ms(void) {
    FILETIME t;ULARGE_INTEGER u;GetSystemTimeAsFileTime(&t);
    u.LowPart=t.dwLowDateTime;u.HighPart=t.dwHighDateTime;
    return u.QuadPart/10000-UINT64_C(11644473600000);
}
int main(void) {
    DWORD pid=0;
    HANDLE game=th10_process(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,&pid);
    if(!game)return 2;
    FILE *out=fopen("Z:\\capture\\rich.bin","wb");
    if(!out)return 3;
    fwrite("TH10OBS1",1,8,out);
    uint64_t start=GetTickCount64();
    while(GetFileAttributesA("Z:\\capture\\actor-closed")==INVALID_FILE_ATTRIBUTES &&
          GetTickCount64()-start<45*60*1000) {
        struct Th10State s;uint64_t t0=utc_ms();uint32_t input,power;
        if(th10_read_state(game,&s) && th10_read(game,CURRENT_NATIVE_INPUT,&input,sizeof(input))
                && th10_read(game,TH10_SCORE+4,&power,sizeof(power))) {
            int count=projectiles(game,s.xy);
            if(count>=0 && s.stage>=1 && s.lives>=-1) {
                uint64_t t1=utc_ms();
                int32_t meta[7]={(int)s.stage,(int)s.difficulty,s.lives,(int)power,(int)s.score,(int)input,count};
                fwrite(&t0,8,1,out);fwrite(&t1,8,1,out);fwrite(meta,4,7,out);fwrite(s.xy,4,2,out);
                float rows[CAPACITY][4]={{0}};
                for(int i=0;i<count&&i<CAPACITY;i++){
                    memcpy(rows[i],objects[i].relative,2*sizeof(float));memcpy(rows[i]+2,objects[i].velocity,2*sizeof(float));
                }
                fwrite(rows,sizeof(rows),1,out);
                if(s.lives<0)break;
            }
        }
        Sleep(38);
    }
    fclose(out);CloseHandle(game);return 0;
}
