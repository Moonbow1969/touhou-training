/* Passive only: sample known player coordinates and the player's matching collision bounds.
   Valid only on original TH13 Trial 0.01a with the documented binary SHA. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int copy(HANDLE p,uint32_t a,void*dst,size_t n){SIZE_T got=0;return ReadProcessMemory(p,(void*)(uintptr_t)a,dst,n,&got)&&got==n;}
int main(void){
 HANDLE list=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);PROCESSENTRY32 e={0};e.dwSize=sizeof e;DWORD pid=0;
 if(Process32First(list,&e))do{if(!_stricmp(e.szExeFile,"th13.exe"))pid=e.th32ProcessID;}while(Process32Next(list,&e));CloseHandle(list);
 HANDLE p=OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);if(!p)return 2;
 FILE*f=fopen("Z:/capture/xy-latest.bin","wb+");if(!f)return 3;
 for(int32_t i=1;;i++){
  uint32_t ptr=0,stat[28]={0};int32_t r[15]={0};float xyz[3],bounds[6];FILETIME ft;GetSystemTimeAsFileTime(&ft);ULARGE_INTEGER u;u.LowPart=ft.dwLowDateTime;u.HighPart=ft.dwHighDateTime;
  if(!copy(p,0x4bcb20,&ptr,4)||!copy(p,0x4b9048,stat,sizeof stat))return 4;
  if(ptr<65536||!copy(p,ptr+0x550,xyz,sizeof xyz)||!copy(p,ptr+0x5a0,bounds,sizeof bounds)){Sleep(6);continue;}
  r[0]=i;r[1]=(int32_t)(u.QuadPart/10000);r[2]=(int32_t)stat[27];r[3]=(int32_t)stat[24];r[4]=(int32_t)stat[14];r[5]=(int32_t)stat[17];
  memcpy(r+6,xyz,8);memcpy(r+8,bounds,8);memcpy(r+10,bounds+3,8);copy(p,0x4df218,r+12,4);copy(p,0x4d9960,r+13,4);r[14]=i;
  rewind(f);if(fwrite(r,sizeof r,1,f)!=1||fflush(f))return 5;Sleep(7);
 }
}
