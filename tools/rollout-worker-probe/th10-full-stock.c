/* Original TH10 trial stock and actual consumed keys, separate from actor inputs. */
#include <inttypes.h>
#include <stdio.h>
#include "th10-trial-state.h"

static uint64_t utc_ms(void) {
    FILETIME t;
    ULARGE_INTEGER u;
    GetSystemTimeAsFileTime(&t);
    u.LowPart=t.dwLowDateTime; u.HighPart=t.dwHighDateTime;
    return u.QuadPart/10000-UINT64_C(11644473600000);
}

static void stopped(const char *reason) {
    FILE *f=fopen("Z:\\capture\\native-stop.tmp","w");
    if(f) {fprintf(f,"%s\n",reason);fclose(f);rename("Z:\\capture\\native-stop.tmp","Z:\\capture\\native-stop");}
}

int main(int argc,char **argv) {
    DWORD pid=0;
    HANDLE game=th10_process(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,&pid);
    if(!game)return 2;
    if(argc==2 && !strcmp(argv[1],"--endpoint")) {
        uint32_t stage,power;int32_t spare;
        int ok=th10_read(game,TH10_STAGE,&stage,sizeof(stage)) &&
               th10_read(game,TH10_TITLE_FIELDS+8,&spare,sizeof(spare)) &&
               th10_read(game,TH10_SCORE+4,&power,sizeof(power));
        if(ok)printf("%" PRIu64 ",%" PRIu32 ",%" PRId32 ",%" PRIu32 "\n",utc_ms(),stage,spare,power);
        CloseHandle(game);
        return ok?0:2;
    }
    puts("utc_ms,stage,difficulty,lives,x,y,raw_score,native_input,power_units,legacy_start_field");
    const uint64_t start=GetTickCount64();uint64_t last=start;
    int status=2;
    const char *reason="watchdog_or_actor_closed";
    while(GetTickCount64()-start<45*60*1000) {
        uint32_t power,input,legacy;
        struct Th10State s;
        if(th10_read_state(game,&s)&&th10_read(game,TH10_SCORE+4,&power,sizeof(power))&&
            th10_read(game,TH10_LEGACY_START_FIELD,&legacy,sizeof(legacy))&&
            th10_read(game,0x4957b4,&input,sizeof(input))) {
            last=GetTickCount64();
            printf("%" PRIu64 ",%" PRIu32 ",%" PRIu32 ",%" PRId32 ",%.2f,%.2f,%" PRIu32 ",%" PRIu32,
                utc_ms(),s.stage,s.difficulty,s.lives,s.xy[0],s.xy[1],s.score,input);
            printf(",%" PRIu32 ",%" PRIu32 "\n",power,legacy);fflush(stdout);
            if(s.lives<0){reason="native_final_life_lost";status=0;break;}
        } else if (GetTickCount64()-last>8000) {
            reason="native_state_unavailable";break;
        }
        if(GetFileAttributesA("Z:\\capture\\actor-closed")!=INVALID_FILE_ATTRIBUTES)break;
        Sleep(25);
    }
    stopped(reason);
    CloseHandle(game);
    return status;
}
