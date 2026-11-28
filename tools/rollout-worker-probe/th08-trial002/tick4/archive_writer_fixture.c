/* Offline PE fixture. Reads a previously published engineering record, no game. */
#define AW_FIXTURE
#include "archive_writer.h"
#include <stdlib.h>

static ArchiveWriter writer;
static uint16_t reference[AW_WORDS], scratch[AW_WORDS];

static int match_file(const char *dir,unsigned stage,unsigned count,const uint16_t *expected){
    char name[AW_PATH];snprintf(name,sizeof(name),"%s/native-record-stage-%u.bin",dir,stage);
    FILE *f=fopen(name,"rb");unsigned header=0;
    if(!f)return 0;
    int ok=fread(&header,4,1,f)==1 && header==count &&
           fread(scratch,2,count,f)==count && memcmp(expected,scratch,2*count)==0 &&
           fgetc(f)==EOF && !ferror(f);
    fclose(f);return ok;
}

int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: fixture new-directory old-stage1.bin ok|short|eof|error\n");return 2;}
    int good=strcmp(argv[3],"ok")==0, force_error=strcmp(argv[3],"error")==0;
    FILE *f=fopen(argv[2],"rb");unsigned count=0;
    if(!f||fread(&count,4,1,f)!=1||count<96||count>AW_WORDS||
       fread(reference,2,count,f)!=count||fgetc(f)!=EOF)return 3;
    fclose(f);
    if(!CreateDirectoryA(argv[1],NULL) || !aw_open(&writer,argv[1]))return 4;
    writer.entered=CreateEventA(NULL,TRUE,FALSE,NULL);
    writer.unblock=CreateEventA(NULL,TRUE,FALSE,NULL);
    if(!writer.entered||!writer.unblock)return 5;
    if(force_error){
        char p[AW_PATH];aw_name(&writer,p,1,"partial");
        if(!CreateDirectoryA(p,NULL))return 6;
    }
    if(!aw_offer(&writer,0,32,reference)||WaitForSingleObject(writer.entered,5000)!=WAIT_OBJECT_0)return 7;
    LARGE_INTEGER t0,t1,freq;QueryPerformanceFrequency(&freq);QueryPerformanceCounter(&t0);
    for(unsigned k=64;k<count;k+=32){
        memcpy(scratch,reference,2*k);
        if(!aw_offer(&writer,0,k,scratch))return 8;
        memset(scratch,0xcc,2*k);
    }
    if(!aw_offer(&writer,0,count,reference))return 9;
    uint16_t stage2[65]={0,123},stage3[1]={456};
    if(!aw_offer(&writer,1,64,stage2)||!aw_offer(&writer,2,1,stage3))return 10;
    QueryPerformanceCounter(&t1);
    /* The original cannot wait for the disk here. The release event is still unset. */
    if(WaitForSingleObject(writer.unblock,0)!=WAIT_TIMEOUT)return 11;
    SetEvent(writer.unblock);
    if(!force_error){
        /* Let the worker persist stage 2 before altering one of its earlier words. */
        char path[AW_PATH];aw_name(&writer,path,2,"partial");
        int ready=0;
        for(int i=0;i<300&&!ready;i++){
            HANDLE h=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,OPEN_EXISTING,0,NULL);
            if(h!=INVALID_HANDLE_VALUE){DWORD n=0;ready=aw_read(h,&n,4)&&n==64;CloseHandle(h);}
            if(!ready)Sleep(10);
        }
        if(!ready)return 12;
        stage2[0]=9876;
        if(!aw_offer(&writer,1,65,stage2))return 13;
    }
    unsigned counts[3]={count,65,1};
    if(strcmp(argv[3],"short")==0)counts[0]++;
    const unsigned *expected=strcmp(argv[3],"eof")==0?NULL:counts;
    int result=aw_finish(&writer,expected);
    if(good!=result)return 14;
    char marker[AW_PATH];aw_name(&writer,marker,0,"tsv");
    if((GetFileAttributesA(marker)!=INVALID_FILE_ATTRIBUTES)!=good)return 15;
    if(good && (!match_file(argv[1],1,count,reference)||!match_file(argv[1],2,65,stage2)||
                !match_file(argv[1],3,1,stage3)))return 16;
    printf("fixture=%s words=%u coalesced=%u writes=%u spoolbytes=%llu offered_while_blocked_us=%.1f sealed=%d\n",
           argv[3],count,writer.coalesced,writer.writes,(unsigned long long)writer.bytes_written,
           1e6*(double)(t1.QuadPart-t0.QuadPart)/freq.QuadPart,result);
    CloseHandle(writer.entered);CloseHandle(writer.unblock);return 0;
}
