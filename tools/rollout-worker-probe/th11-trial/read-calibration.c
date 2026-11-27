/* Engineering calibration in the local, unmodified original. A request records
   only its .data section; the final native worker uses verified fields alone. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    PROCESSENTRY32 pe={0};pe.dwSize=sizeof(pe);DWORD pid=0;HANDLE list=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    if(Process32First(list,&pe))do {if(!_stricmp(pe.szExeFile,"th11.exe")||!_stricmp(pe.szExeFile,"th11tr.exe")){if(pid)return 2;pid=pe.th32ProcessID;}}while(Process32Next(list,&pe));CloseHandle(list);
    if(!pid)return 3;MODULEENTRY32 module={0};module.dwSize=sizeof(module);list=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);if(!Module32First(list,&module))return 4;CloseHandle(list);
    HANDLE process=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);if(!process)return 5;
    SIZE_T n=0;unsigned char image[4096];uintptr_t base=(uintptr_t)module.modBaseAddr;
    if(!ReadProcessMemory(process,(void*)base,image,sizeof(image),&n)||n!=sizeof(image))return 6;
    IMAGE_NT_HEADERS32 *nt=(IMAGE_NT_HEADERS32*)(image+((IMAGE_DOS_HEADER*)image)->e_lfanew);IMAGE_SECTION_HEADER *s=IMAGE_FIRST_SECTION(nt);
    int index=-1;for(int j=0;j<nt->FileHeader.NumberOfSections;j++)if(!strncmp((char*)s[j].Name,".data",5))index=j;
    if(index<0)return 7;uintptr_t rva=s[index].VirtualAddress;size_t length=s[index].Misc.VirtualSize;if(!length||length>4194304)return 8;
    unsigned char *data=malloc(length);if(!data)return 9;FILE *meta=fopen("Z:/capture/calibration-data.txt","w");fprintf(meta,"%lu %lu %lu\n",(unsigned long)base,(unsigned long)rva,(unsigned long)length);fclose(meta);
    int prior=-1;for(int i=0;i<170000;i++){
        FILE *req=fopen("Z:/capture/calibration-request.txt","r");int id=-1;if(req){fscanf(req,"%d",&id);fclose(req);}
        if(id>=0&&id!=prior){
            char path[160];if(!ReadProcessMemory(process,(void*)(base+rva),data,length,&n)||n!=length)break;
            snprintf(path,sizeof(path),"Z:/capture/calibration-%03d.bin",id);FILE *out=fopen(path,"wb");if(!out)return 10;
            fwrite(data,1,length,out);fclose(out);prior=id;
        }
        if(id==-2)break;Sleep(5);
    }
    free(data);CloseHandle(process);return 0;
}
