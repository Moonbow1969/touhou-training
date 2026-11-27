/* TH17 0.01a read-only, 3ffa3963...7aaba. Native values are excluded from the model. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

typedef struct {
    int32_t stage, start_stage, tick, level_tick, total_tick, character, beast, score, rank, continues,
            power, lives, life_pieces, bombs, bomb_pieces;
} State;
typedef struct { volatile LONG sequence; State state; } Shared;

int main(void) {
    DWORD pid=0; PROCESSENTRY32 pe={.dwSize=sizeof pe};
    HANDLE ss=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    if(Process32First(ss,&pe))do{if(!_stricmp(pe.szExeFile,"th17.exe")){if(pid)return 2;pid=pe.th32ProcessID;}}while(Process32Next(ss,&pe));
    CloseHandle(ss);if(!pid)return 3;
    HANDLE game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if(!game)return 4;
    HANDLE file=CreateFileA("live.state",GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,CREATE_NEW,0,NULL);
    if(file==INVALID_HANDLE_VALUE)return 5;
    HANDLE mapping=CreateFileMapping(file,NULL,PAGE_READWRITE,0,sizeof(Shared),NULL);
    Shared *shared=MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,0);
    FILE *log=fopen("native.tsv","wb");if(!shared||!log)return 6;
    fprintf(log,"ms\tstage\tstart_stage\ttick\tlevel_tick\ttotal_tick\tcharacter\tbeast\tscore\trank\tcontinues\tpower\tspares\tlife_pieces\tbombs\tbomb_pieces\n");
    DWORD begin=GetTickCount();int last_tick=-1,armed=0,drained=0;
#ifdef TH17_COMPLETE_STATE_LOG
    State last_state={0};int have_previous=0;
#endif
    for(;;){
        int32_t v[31];SIZE_T n=0;
        if(!ReadProcessMemory(game,(void*)0x4b1634,v,sizeof v,&n)||n!=sizeof v)return 7;
        State s={v[0],v[1],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[21],v[25],v[26],v[28],v[29]};
        InterlockedIncrement(&shared->sequence);shared->state=s;InterlockedIncrement(&shared->sequence);
        if(s.stage==1&&s.lives==2&&s.bombs==3&&s.tick>=0&&s.tick<200)armed=1;
#ifdef TH17_COMPLETE_STATE_LOG
        /* Stage, inventory and difficulty can change while the BGM timer is stopped. */
        int changed=!have_previous || memcmp(&s,&last_state,sizeof s)!=0;
        last_state=s;have_previous=1;
#else
        int changed=last_tick!=s.total_tick;
#endif
        if(changed || !drained && armed && s.lives<0){
            fprintf(log,"%lu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",(unsigned long)(GetTickCount()-begin),
                s.stage,s.start_stage,s.tick,s.level_tick,s.total_tick,s.character,s.beast,s.score,s.rank,s.continues,s.power,s.lives,s.life_pieces,s.bombs,s.bomb_pieces);
            fflush(log);last_tick=s.total_tick;
        }
        if(armed&&s.lives<0&&++drained>90)break;
        if(GetTickCount()-begin>3600000)return 8;
        Sleep(10);
    }
    fclose(log);UnmapViewOfFile(shared);CloseHandle(mapping);CloseHandle(file);CloseHandle(game);return 0;
}
