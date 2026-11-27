/* Local official 0.02a state, sampled for validation and credit endpoints. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    PROCESSENTRY32 entry={0};entry.dwSize=sizeof(entry);
    HANDLE list=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    DWORD pid=0;
    if(Process32First(list,&entry)) do {
        if(!_stricmp(entry.szExeFile,"th11.exe")){if(pid)return 2;pid=entry.th32ProcessID;}
    } while(Process32Next(list,&entry));
    CloseHandle(list);
    if(!pid)return 3;
    MODULEENTRY32 module={0};module.dwSize=sizeof(module);
    list=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    if(!Module32First(list,&module))return 4;
    CloseHandle(list);
    HANDLE game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if(!game)return 5;
    uintptr_t base=(uintptr_t)module.modBaseAddr;
    FILE *file=fopen("Z:/capture/native.bin","wb");
    if(!file)return 6;
    for(int i=0;i<240000;i++){
        if(GetFileAttributesA("Z:/capture/stop-stock")!=INVALID_FILE_ATTRIBUTES)break;
        int32_t globals[30],key,record[12];SIZE_T n=0;
        if(!ReadProcessMemory(game,(void*)(base+0x9e338),globals,sizeof(globals),&n)||n!=sizeof(globals))break;
        if(!ReadProcessMemory(game,(void*)(base+0xc1ef0),&key,4,&n)||n!=4)break;
        record[0]=GetTickCount();record[1]=globals[1];record[2]=globals[2];record[3]=globals[12];
        record[4]=globals[13];record[5]=globals[14];record[6]=globals[16];record[7]=globals[18];
        record[8]=globals[23];record[9]=globals[27];record[10]=globals[29];record[11]=key;
        if(fwrite(record,sizeof(record),1,file)!=1)break;
        fflush(file);Sleep(10);
    }
    fclose(file);CloseHandle(game);return 0;
}
