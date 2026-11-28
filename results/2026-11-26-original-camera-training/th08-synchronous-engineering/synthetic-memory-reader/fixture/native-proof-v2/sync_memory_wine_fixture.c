#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static volatile unsigned long value;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 if(!strcmp(argv[1],"target")){
  HANDLE file=CreateFileA("Z:\\capture\\tick.bin",GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
  HANDLE section=CreateFileMappingA(file,NULL,PAGE_READWRITE,0,4096,NULL);
  volatile unsigned long *shared=MapViewOfFile(section,FILE_MAP_ALL_ACCESS,0,0,4096); if(!shared)return 3;
  FILE *p=fopen("Z:\\capture\\target.txt","w");fprintf(p,"%lu %lu\n",GetCurrentProcessId(),(unsigned long)(uintptr_t)&value);fclose(p);
  for(;;){*shared=++value;Sleep(1);}
 }
 FILE *p=fopen("Z:\\capture\\target.txt","r");unsigned long pid,addr;
 if(!p||fscanf(p,"%lu %lu",&pid,&addr)!=2)return 4;
 fclose(p);HANDLE h=OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid); if(!h)return 5;
 FILE *ok=fopen("Z:\\capture\\reader-ready","w");fputs("ready\n",ok);fclose(ok);
 for(;;){for(int i=0;i<12;i++){unsigned long dest;SIZE_T n;if(!ReadProcessMemory(h,(void*)(uintptr_t)addr,&dest,sizeof(dest),&n)||n!=sizeof(dest))return 6;}Sleep(1);}
}
