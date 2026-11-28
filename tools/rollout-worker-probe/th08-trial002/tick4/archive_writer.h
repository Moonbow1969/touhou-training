/* Single-producer, single-writer archive for one original TH08 credit.
 * Only the producer touches submitted[]. The worker never holds lock during I/O.
 * Unsealed partials are diagnostic and never count as complete stage evidence. */
#ifndef TH08_ARCHIVE_WRITER_H
#define TH08_ARCHIVE_WRITER_H
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define AW_STAGES 3
#define AW_WORDS 65534u
#define AW_PATH 300

typedef struct {
    CRITICAL_SECTION lock;
    HANDLE ready, thread, file[AW_STAGES];
    char directory[AW_PATH];
    uint16_t pending[AW_STAGES][AW_WORDS], cached[AW_STAGES][AW_WORDS], work[AW_WORDS];
    unsigned queued[AW_STAGES], submitted[AW_STAGES], stored[AW_STAGES];
    unsigned next, writes, coalesced;
    uint64_t bytes_written;
    LONG closing, error;
#ifdef AW_FIXTURE
    HANDLE entered, unblock;
#endif
} ArchiveWriter;

static void aw_name(const ArchiveWriter *w, char path[AW_PATH], unsigned stage, const char *suffix){
    if(stage) snprintf(path,AW_PATH,"%s/native-record-stage-%u.%s",w->directory,stage,suffix);
    else snprintf(path,AW_PATH,"%s/native-archive-sealed.%s",w->directory,suffix);
}
static int aw_seek(HANDLE f,unsigned offset){
    return SetFilePointer(f,(LONG)offset,NULL,FILE_BEGIN)!=INVALID_SET_FILE_POINTER;
}
static int aw_write(HANDLE f,const void *buf,DWORD n){
    DWORD actual=0;
    return WriteFile(f,buf,n,&actual,NULL) && actual==n;
}
static int aw_read(HANDLE f,void *buf,DWORD n){
    DWORD actual=0;
    return ReadFile(f,buf,n,&actual,NULL) && actual==n;
}
static int aw_store(ArchiveWriter *w,unsigned s,unsigned count){
    char path[AW_PATH];
    unsigned old=w->stored[s];
    if(!w->file[s]){
        aw_name(w,path,s+1,"partial");
        HANDLE f=CreateFileA(path,GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ,NULL,
                             CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL);
        if(f==INVALID_HANDLE_VALUE)return 0;
        w->file[s]=f;
    }
    HANDLE f=w->file[s];
    unsigned keep=memcmp(w->cached[s],w->work,2u*old)==0?old:0u;
    unsigned n=2u*(count-keep);
    if(!aw_seek(f,4u+2u*keep)||!aw_write(f,w->work+keep,n)||
       !aw_seek(f,0)||!aw_write(f,&count,4))return 0;
    memcpy(w->cached[s],w->work,2u*count);
    w->stored[s]=count;
    w->bytes_written+=4u+n;
    w->writes++;
    return 1;
}
static DWORD WINAPI aw_thread(LPVOID arg){
    ArchiveWriter *w=(ArchiveWriter*)arg;
    for(;;){
        unsigned s=0,count=0;
        EnterCriticalSection(&w->lock);
        for(unsigned i=0;i<AW_STAGES;i++){
            s=(w->next+i)%AW_STAGES;
            if(w->queued[s]){
                count=w->queued[s];w->queued[s]=0;w->next=(s+1)%AW_STAGES;
                memcpy(w->work,w->pending[s],2u*count);
                break;
            }
        }
        int finish=w->closing&&!count;
        LeaveCriticalSection(&w->lock);
        if(finish)return 0;
        if(!count){
            if(WaitForSingleObject(w->ready,INFINITE)==WAIT_OBJECT_0)continue;
        }else{
#ifdef AW_FIXTURE
            if(w->entered){SetEvent(w->entered);WaitForSingleObject(w->unblock,INFINITE);}
#endif
            if(aw_store(w,s,count))continue;
        }
        DWORD code=GetLastError();
        InterlockedExchange(&w->error,(LONG)(code?code:ERROR_WRITE_FAULT));
        return 1;
    }
}
static int aw_open(ArchiveWriter *w,const char *directory){
    if(strlen(directory)+48>=AW_PATH)return 0;
    strcpy(w->directory,directory);
    /* Immutable output ownership, including retries after partial failure. */
    for(unsigned i=0;i<=AW_STAGES;i++){
        char path[AW_PATH];aw_name(w,path,i,i?"bin":"tsv");
        if(GetFileAttributesA(path)!=INVALID_FILE_ATTRIBUTES)return 0;
        aw_name(w,path,i,i?"partial":"tmp");
        if(GetFileAttributesA(path)!=INVALID_FILE_ATTRIBUTES)return 0;
    }
    InitializeCriticalSection(&w->lock);
    w->ready=CreateEventA(NULL,FALSE,FALSE,NULL);
    if(!w->ready){DeleteCriticalSection(&w->lock);return 0;}
    w->thread=CreateThread(NULL,0,aw_thread,w,0,NULL);
    if(!w->thread){CloseHandle(w->ready);DeleteCriticalSection(&w->lock);return 0;}
    return 1;
}
static int aw_failed(ArchiveWriter *w){return InterlockedCompareExchange(&w->error,0,0)!=0;}
static int aw_offer(ArchiveWriter *w,unsigned s,unsigned count,const uint16_t *words){
    if(s>=AW_STAGES||count<=w->submitted[s]||count>AW_WORDS||aw_failed(w))return 0;
    EnterCriticalSection(&w->lock);
    if(w->queued[s])w->coalesced++;
    memcpy(w->pending[s],words,2u*count);
    w->queued[s]=count;w->submitted[s]=count;
    LeaveCriticalSection(&w->lock);
    return SetEvent(w->ready)!=0;
}
/* Join before accessing the worker's stored image. Timeout means the owning
 * process must exit. Never publish a success receipt on that path. */
static int aw_finish(ArchiveWriter *w,const unsigned *expected){
    EnterCriticalSection(&w->lock);w->closing=1;LeaveCriticalSection(&w->lock);
    SetEvent(w->ready);
    if(WaitForSingleObject(w->thread,20000)!=WAIT_OBJECT_0)return 0;
    int ok=expected&&!aw_failed(w);
    unsigned total=0;
    for(unsigned s=0;s<AW_STAGES;s++){
        if(expected && expected[s]!=w->stored[s])ok=0;
        total+=w->stored[s];
        if(!w->file[s])continue;
        if(ok){
            DWORD count=0;
            HANDLE f=w->file[s];
            if(!FlushFileBuffers(f)||GetFileSize(f,NULL)!=4u+2u*w->stored[s]||
               !aw_seek(f,0)||!aw_read(f,&count,4)||count!=w->stored[s]||
               !aw_read(f,w->work,2u*count)||memcmp(w->cached[s],w->work,2u*count))ok=0;
        }
        if(!CloseHandle(w->file[s]))ok=0;
    }
    CloseHandle(w->thread);CloseHandle(w->ready);DeleteCriticalSection(&w->lock);
    if(!ok||!total)return 0;
    char a[AW_PATH],b[AW_PATH];
    for(unsigned s=0;s<AW_STAGES;s++)if(w->stored[s]){
        aw_name(w,a,s+1,"partial");aw_name(w,b,s+1,"bin");
        if(!MoveFileExA(a,b,MOVEFILE_WRITE_THROUGH))return 0;
    }
    aw_name(w,a,0,"tmp");aw_name(w,b,0,"tsv");
    HANDLE f=CreateFileA(a,GENERIC_WRITE,0,NULL,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL);
    if(f==INVALID_HANDLE_VALUE)return 0;
    char row[220];
    int n=snprintf(row,sizeof(row),"stage1\tstage2\tstage3\twrites\tcoalesced\tspool_write_bytes\n%u\t%u\t%u\t%u\t%u\t%llu\n",
                   w->stored[0],w->stored[1],w->stored[2],w->writes,w->coalesced,(unsigned long long)w->bytes_written);
    ok=n>0 && (size_t)n<sizeof(row) && aw_write(f,row,(DWORD)n) && FlushFileBuffers(f);
    if(!CloseHandle(f))ok=0;
    return ok && MoveFileExA(a,b,MOVEFILE_WRITE_THROUGH);
}
#endif
