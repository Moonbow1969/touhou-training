/* Read-only original TH12 0.02a telemetry, isolated from the policy and input control. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int32_t read_i32(HANDLE game,uintptr_t address) {
    int32_t v=-1000; SIZE_T n=0;
    return ReadProcessMemory(game,(const void *)address,&v,4,&n) && n==4?v:-1000;
}
int main(int argc,char **argv) {
#ifdef TH12_TICK_MMAP
    if(argc!=4)return 2;
#else
    if(argc!=3)return 2;
#endif
    DWORD pid=0; PROCESSENTRY32 pe={0};pe.dwSize=sizeof pe;
    HANDLE ss=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    if(Process32First(ss,&pe))do{
        if(!_stricmp(pe.szExeFile,"th12.exe")){if(pid)return 3;pid=pe.th32ProcessID;}
    }while(Process32Next(ss,&pe)); CloseHandle(ss);
    HANDLE game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if(!pid||!game)return 4;
    FILE *out=fopen(argv[1],"wb"); if(!out)return 5;
#ifdef TH12_TICK_MMAP
    struct TickState { LONG sequence,stage,manager_stage,tick,mode,keys,spares,bombs; };
    HANDLE path=CreateFileA(argv[3],GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
    if(path==INVALID_HANDLE_VALUE)return 9;
    HANDLE mapping=CreateFileMappingA(path,NULL,PAGE_READWRITE,0,sizeof(struct TickState),NULL);
    volatile struct TickState *shared=MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(struct TickState));
    if(!shared)return 9;
#ifdef TH12_FIELD_XY
    fprintf(out,"ms\tstage\tmanager_stage\ttick\tmode\tspares\tbombs\tkeys\tx_fixed\ty_fixed\tx_float_bits\ty_float_bits\n");
#else
    fprintf(out,"ms\tstage\tmanager_stage\ttick\tmode\tspares\tbombs\tkeys\n");
#endif
    LONG last_keys=-1;
#else
    fprintf(out,"ms\tstage\tmanager_stage\ttick\tmode\tspares\tbombs\n");
#endif
    int32_t previous[6]={-1000,-1000,-1000,-1000,-1000,-1000};
    DWORD begin=GetTickCount(),last_write=begin;
    while(GetTickCount()-begin<3600000){
        int32_t m=read_i32(game,0x4b21d4);
        int32_t row[6]={read_i32(game,0x4aea38),-1,-1,-1,read_i32(game,0x4aea20),read_i32(game,0x4aea28)};
        if(m>65536){
            row[1]=read_i32(game,(uintptr_t)m+0x108);
            row[2]=read_i32(game,(uintptr_t)m+0x100);
            row[3]=read_i32(game,(uintptr_t)m+0x10);
        }
        if(row[0]!=read_i32(game,0x4aea38) || m!=read_i32(game,0x4b21d4))continue;
#ifdef TH12_TICK_MMAP
        int32_t keys=read_i32(game,0x4d2670);
        if(m>65536 && row[2]!=read_i32(game,(uintptr_t)m+0x100))continue;
        InterlockedIncrement(&shared->sequence);
        shared->stage=row[0];shared->manager_stage=row[1];shared->tick=row[2];shared->mode=row[3];
        shared->keys=keys;shared->spares=row[4];shared->bombs=row[5];
        InterlockedIncrement(&shared->sequence);
#endif
        DWORD now=GetTickCount();
        int change=row[0]!=previous[0]||row[1]!=previous[1]||row[3]!=previous[3]||
                   row[4]!=previous[4]||row[5]!=previous[5]||row[2]/8!=previous[2]/8||now-last_write>=200;
        int stopped=GetFileAttributesA(argv[2])!=INVALID_FILE_ATTRIBUTES;
#ifdef TH12_TICK_MMAP
        change=change || row[2]!=previous[2] || keys!=last_keys;
        if(change||stopped){
#ifdef TH12_FIELD_XY
            int32_t player=read_i32(game,0x4b21d0);
            if(player<65536)return 11;
            fprintf(out,"%lu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",(unsigned long)(now-begin),row[0],row[1],row[2],row[3],row[4],row[5],keys,
                read_i32(game,(uintptr_t)player+0x988),read_i32(game,(uintptr_t)player+0x98c),read_i32(game,(uintptr_t)player+0x97c),read_i32(game,(uintptr_t)player+0x980));
#else
            fprintf(out,"%lu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",(unsigned long)(now-begin),
                    row[0],row[1],row[2],row[3],row[4],row[5],keys);
#endif
            fflush(out);memcpy(previous,row,sizeof row);last_write=now;last_keys=keys;
        }
        if(stopped){UnmapViewOfFile((void *)shared);CloseHandle(mapping);CloseHandle(path);fclose(out);CloseHandle(game);return 0;}
        Sleep(1);
#else
        if(change||stopped){
            fprintf(out,"%lu\t%d\t%d\t%d\t%d\t%d\t%d\n",(unsigned long)(now-begin),
                    row[0],row[1],row[2],row[3],row[4],row[5]);
            fflush(out);memcpy(previous,row,sizeof row);last_write=now;
        }
        if(stopped){fclose(out);CloseHandle(game);return 0;}
        Sleep(8);
#endif
    }
    fclose(out);CloseHandle(game);return 6;
}

