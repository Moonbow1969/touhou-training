#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
    while (getchar() == 'q') {
        uint32_t obj=0,tick=0,end=0,mode=~0u,info=0,origin=0,cursor=0,stage=99,flags=0;
        uint16_t keys=0,raw=0,prev=0,buf[2]={0,0};
        float xy[2]={0,0}; uint8_t player[4]={0,0,0,0};
        copy_read(game,base+0x14a9340,&obj,4);
        int valid = copy_read(game,obj,&tick,4);
        copy_read(game,base+0x120c760,&stage,4);
        copy_read(game,obj+0x10,&mode,4);
        copy_read(game,obj+8,&info,4);
        copy_read(game,obj+0x50,&cursor,4);
        if(info && stage<3) copy_read(game,info+0x20+stage*4,&origin,4);
        int buffer_read = cursor>2 && copy_read(game,cursor-2,buf,4);
        copy_read(game,base+0x120c54c,&flags,4);
        copy_read(game,base+0x120c9b8,&raw,2);
        copy_read(game,base+0x120c9bc,&keys,2);
        copy_read(game,base+0x120c9c4,&prev,2);
        copy_read(game,base+0x13cc810,player,4);
        copy_read(game,base+0x13ccac4,xy,8);
        valid = valid && copy_read(game,obj,&end,4) && tick == end;
        static unsigned saved_tick[3];
        if(valid && mode==0 && flags==4 && stage<3 && tick>=saved_tick[stage]+8 && tick<65535 &&
           cursor==origin+36+2*tick && origin) {
            static uint16_t payload[65535];
            if(copy_read(game,origin+36,payload,2*tick)){
                char filename[96];
                snprintf(filename,sizeof(filename),"Z:/capture/native-record-stage-%u.bin",stage+1);
                FILE *dump=fopen("Z:/capture/native-record.tmp","wb");
                if(dump){fwrite(&tick,4,1,dump);fwrite(payload,2,tick,dump);fclose(dump);
                    if(MoveFileExA("Z:/capture/native-record.tmp",filename,MOVEFILE_REPLACE_EXISTING))
                        saved_tick[stage]=tick;
                }
            }
        }
        printf("%lu %lu %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %.4f %.4f %d %d\n",
               pid,(unsigned long)base,obj,stage,tick,mode,flags,raw,keys,prev,info,origin,cursor,
               buf[0],buf[1],player[0],player[3],xy[0],xy[1],buffer_read,valid);
    }
    CloseHandle(game);
    return 0;
}
