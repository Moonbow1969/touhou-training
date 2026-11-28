#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "archive_writer.h"
static ArchiveWriter archive;

static int copy_read(HANDLE handle, uintptr_t address, void *value, size_t size) {
    SIZE_T got = 0;
    return address && ReadProcessMemory(handle, (void *)address, value, size, &got) && got == size;
}

int main(void) {
    PROCESSENTRY32 entry = { .dwSize = sizeof(entry) };
    HANDLE list = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    DWORD pid = 0;
    if (Process32First(list, &entry)) do {
        if (!_stricmp(entry.szExeFile, "th08tr.exe")) { pid = entry.th32ProcessID; break; }
    } while (Process32Next(list, &entry));
    CloseHandle(list);
    MODULEENTRY32 mod = { .dwSize = sizeof(mod) };
    list = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid);
    if (!pid || !Module32First(list, &mod)) return 2;
    CloseHandle(list);
    HANDLE game = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!game) return 3;
    uintptr_t base = (uintptr_t)mod.modBaseAddr;
    setvbuf(stdout, NULL, _IONBF, 0);
    if(!aw_open(&archive,"Z:/capture")){fprintf(stderr,"archive open failed %lu\n",GetLastError());return 10;}
    int command;
    while ((command=getchar()) == 'q') {
        if(aw_failed(&archive)){fprintf(stderr,"archive worker error %ld\n",archive.error);break;}
        uint32_t obj=0,tick=0,end=0,mode=~0u,info=0,origin=0,cursor=0,stage=99,flags=0;
        uint16_t keys=0,raw=0,prev=0,buf[2]={0,0};
        float xy[2]={0,0}; uint8_t player[4]={0,0,0,0};
        copy_read(game,base+0x14a9340,&obj,4);
        uint8_t recording[0x54],globals[0x47a],state[0x2bc];
        int valid=copy_read(game,obj,recording,sizeof(recording));
        if(valid){
            memcpy(&tick,recording,4);memcpy(&mode,recording+0x10,4);
            memcpy(&info,recording+8,4);memcpy(&cursor,recording+0x50,4);
        }
        int shared_ok=copy_read(game,base+0x120c54c,globals,sizeof(globals));
        if(shared_ok){
            memcpy(&flags,globals,4);memcpy(&stage,globals+0x214,4);
            memcpy(&raw,globals+0x46c,2);memcpy(&keys,globals+0x470,2);memcpy(&prev,globals+0x478,2);
        }
        if(info && stage<3) copy_read(game,info+0x20+stage*4,&origin,4);
        int buffer_read = cursor>2 && copy_read(game,cursor-2,buf,4);
        if(copy_read(game,base+0x13cc810,state,sizeof(state))){
            memcpy(player,state,4);memcpy(xy,state+0x2b4,8);
        }
        valid = valid && shared_ok && copy_read(game,obj,&end,4) && tick == end;
        unsigned count=tick+(flags==0);
        if(valid && mode==0 && stage<3 && ((flags==4 && count>=archive.submitted[stage]+32) ||
           (flags==0 && count>archive.submitted[stage])) && count<=AW_WORDS &&
           cursor==origin+36+2*tick && origin) {
            static uint16_t payload[AW_WORDS];
            uint32_t after_tick, after_stage, after_cursor, after_flags, after_origin, after_obj;
            /* Protect a borrowed source against original buffer/stage changes during copy.
             * A failed bracket remains visible through the existing stable bit. */
            valid=copy_read(game,origin+36,payload,2*count) &&
                  copy_read(game,base+0x14a9340,&after_obj,4) && after_obj==obj &&
                  copy_read(game,base+0x120c760,&after_stage,4) && after_stage==stage &&
                  copy_read(game,base+0x120c54c,&after_flags,4) && after_flags==flags &&
                  copy_read(game,info+0x20+stage*4,&after_origin,4) && after_origin==origin &&
                  copy_read(game,obj+0x50,&after_cursor,4) && after_cursor==cursor &&
                  copy_read(game,obj,&after_tick,4) && after_tick==tick;
            if(valid && !aw_offer(&archive,stage,count,payload)){
                fprintf(stderr,"native archive offer failed\n");break;
            }
        }
        printf("%lu %lu %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %.4f %.4f %d %d\n",
               pid,(unsigned long)base,obj,stage,tick,mode,flags,raw,keys,prev,info,origin,cursor,
               buf[0],buf[1],player[0],player[3],xy[0],xy[1],buffer_read,valid);
    }
    CloseHandle(game);
    unsigned counts[3];char tail, line[128];
    int explicit_seal=command=='s' && fgets(line,sizeof(line),stdin) &&
        strchr(line,'\n') && sscanf(line,"%u %u %u %c",&counts[0],&counts[1],&counts[2],&tail)==3;
    int sealed=aw_finish(&archive,explicit_seal?counts:NULL);
    if(!sealed){
        fprintf(stderr,"native archive UNSEALED command=%d error=%ld stored=%u,%u,%u\n",
                command,archive.error,archive.submitted[0],archive.submitted[1],archive.submitted[2]);return 11;
    }
    printf("SEALED %u %u %u\n",counts[0],counts[1],counts[2]);
    return 0;
}
