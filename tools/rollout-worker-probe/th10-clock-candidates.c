/* Read-only engineering search for frame counters in the original TH10 trial globals
   and their top-level managers. Persist just counters which track known native motion. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "th10-trial-state.h"

#define WORDS 1536
#define BLOCKS 175
#define PERIODS 50
struct range {
    uint32_t key, base;
    int indirect, present;
    int32_t old[WORDS],now[WORDS];
    unsigned char good[WORDS],seen[WORDS];
};
static struct range scanned[BLOCKS];
struct result {uint32_t key,offset;int indirect;};
static struct result results[512];

static uint64_t ms(void) {
    FILETIME ft;ULARGE_INTEGER u;GetSystemTimeAsFileTime(&ft);
    u.LowPart=ft.dwLowDateTime;u.HighPart=ft.dwHighDateTime;
    return u.QuadPart/10000-UINT64_C(11644473600000);
}
static unsigned stage_timer(HANDLE h) {
    uint32_t ptr=0,t=0;
    if(th10_read(h,0x475448,&ptr,4))th10_read(h,ptr+0x3c,&t,4);
    return t;
}
int main(void) {
    DWORD pid;HANDLE h=th10_process(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,&pid);
    if(!h)return 2;
    int count=0;
    for(uint32_t p=0x472000;p<0x476000;p+=4*WORDS)scanned[count++].key=p;
    for(uint32_t p=0x494000;p<0x496000;p+=4*WORDS)scanned[count++].key=p;
    for(uint32_t p=0x475400;p<0x475680;p+=4){
        uint32_t v=0;
        if(th10_read(h,p,&v,4)&&v>0x500000&&v<0x7f000000&&!(v%4)){
            scanned[count].key=p;scanned[count].indirect=1;count++;
        }
    }
    int updates=0;int64_t previous=0;
    for(int sample=0;sample<=PERIODS;sample++){
        int64_t now=stage_timer(h);int delta=(int)(now-previous);previous=now;
        int valid=sample>0&&delta>=2&&delta<40;updates+=valid;
        for(int i=0;i<count;i++){
            struct range *b=&scanned[i];uint32_t base=b->key;
            if(b->indirect&&!th10_read(h,b->key,&base,4))continue;
            if(base<=0x10000||!th10_read(h,base,b->now,sizeof(b->now))){b->present=0;continue;}
            if(valid&&b->present&&b->base==base){
                for(int k=0;k<WORDS;k++){
                    if(b->old[k]<0||b->now[k]<0||b->now[k]>200000)continue;
                    int change=b->now[k]-b->old[k];
                    b->seen[k]++;b->good[k]+=abs(change-delta)<=1;
                }
            }
            b->base=base;b->present=1;memcpy(b->old,b->now,sizeof(b->old));
        }
        Sleep(170);
    }
    int n=0;
    for(int i=0;i<count;i++)for(int k=0;k<WORDS;k++){
        struct range *b=&scanned[i];
        if(updates>20&&b->seen[k]>=updates*.9&&b->good[k]>=updates*.92&&n<512)
            results[n++]=(struct result){b->key,(uint32_t)4*k,b->indirect};
    }
    fprintf(stderr,"bounded manager ranges=%d native-motion periods=%d candidates=%d\n",count,updates,n);
    printf("utc_ms,stage,stage_background_tick");
    for(int i=0;i<n;i++)printf(",%c%x_%x",results[i].indirect?'p':'g',results[i].key,results[i].offset);
    putchar('\n');fflush(stdout);
    for(int step=0;step<12000;step++){
        if(GetFileAttributesA("Z:\\capture\\clock-stop")!=INVALID_FILE_ATTRIBUTES)break;
        uint32_t stage=0;th10_read(h,TH10_STAGE,&stage,4);
        printf("%llu,%u,%u",(unsigned long long)ms(),stage,stage_timer(h));
        for(int i=0;i<n;i++){
            uint32_t base=results[i].key;int32_t v;
            if((!results[i].indirect||th10_read(h,results[i].key,&base,4))&&th10_read(h,base+results[i].offset,&v,4))printf(",%d",v);
            else putchar(',');
        }
        putchar('\n');fflush(stdout);Sleep(45);
    }
    CloseHandle(h);return 0;
}
