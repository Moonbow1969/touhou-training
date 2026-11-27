/* Engineering only, pinned TH17 trial. Record game-owned input-state candidates
   around the homologous input-data area, bracketed by two identical stage clocks.
   No value from this observer is sent to the camera actor. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define START 0x4ae000
#define BYTES 0x3000
static int sample(HANDLE game,uintptr_t where,void *value,size_t size){
    SIZE_T n=0;return ReadProcessMemory(game,(void*)where,value,size,&n)&&n==size;
}
int main(void){
    HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    PROCESSENTRY32 pe={.dwSize=sizeof pe};DWORD pid=0;
    if(Process32First(snap,&pe))do{if(!_stricmp(pe.szExeFile,"th17.exe")){if(pid)return 2;pid=pe.th32ProcessID;}}while(Process32Next(snap,&pe));CloseHandle(snap);
    HANDLE game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if(!game)return 3;
    FILE *out=fopen("input-qualification.bin","wb");if(!out)return 4;
    const uint32_t header[]={0x51493731,START,BYTES,16};
    if(fwrite(header,sizeof(header),1,out)!=1)return 5;
    int seen=0,last=-1;DWORD start=GetTickCount();
    while(GetTickCount()-start<150000){
        int32_t a[31],b[31];uint8_t block[BYTES];
        if(!sample(game,0x4b1634,a,sizeof(a)))return 6;
        if(a[0]==1 && (a[9]==2||a[9]==3) && a[3]>0 && a[3]!=last && a[25]>=0){
            if(!sample(game,START,block,sizeof(block))||!sample(game,0x4b1634,b,sizeof(b)))return 7;
            if(a[0]==b[0]&&a[3]==b[3]&&a[25]==b[25]){
                int32_t row[]={a[0],a[3],a[25],a[28]};
                if(fwrite(row,sizeof(row),1,out)!=1||fwrite(block,sizeof(block),1,out)!=1)return 8;
                last=a[3];seen++;
                if(seen>=520)break;
            }
        }
        Sleep(3);
    }
    fclose(out);CloseHandle(game);return seen>=384?0:9;
}
