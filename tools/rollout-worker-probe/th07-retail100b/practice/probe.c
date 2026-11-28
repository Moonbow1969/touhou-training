#define main old_main
#include "native_input.c"
#undef main
static int field(int fd,void *v,size_t n,uint32_t a){return pread(fd,v,n,a)==(ssize_t)n;}
int main(void){
    int fd=open_game();if(fd<0)return 1;
    long long until=mono_ns()+14000000000LL;
    puts("ns,stage,stage_tick,manager_tick,rank,shot,spares,bombs,misses,continues,keys,flags");
    unsigned count=0,last=UINT32_MAX,stage_first=0;
    while(mono_ns()<until){
        uint32_t s=0,m=0,g=0,t=0,tm=0,r=0,t2=0,s2=0,flags=0;
        uint8_t shot=0,c=0;uint16_t k=0;float lives=0,bombs=0,miss=0;
        if(!field(fd,&s,4,0x62f85c)||!field(fd,&m,4,0x4b9e48)||!m||!field(fd,&g,4,0x626278)||!g||
           !field(fd,&t,4,0x62f858)||!field(fd,&tm,4,m)||!field(fd,&r,4,0x626280)||
           !field(fd,&shot,1,0x62f647)||!field(fd,&flags,4,0x62f648)||
           !field(fd,&lives,4,g+0x5c)||!field(fd,&bombs,4,g+0x68)||!field(fd,&miss,4,g+0x50)||
           !field(fd,&c,1,g+0x20)||!field(fd,&k,2,0x4b9e50)||!field(fd,&t2,4,0x62f858)||!field(fd,&s2,4,0x62f85c)||
           s!=s2||t!=t2||!(flags&4)||s<1||s>6||!t||t!=tm||last==t){usleep(1000);continue;}
        if(!count)stage_first=s;
        if(s!=stage_first)return 2;
        printf("%lld,%u,%u,%u,%u,%u,%.0f,%.0f,%.0f,%u,%u,%u\n",mono_ns(),s,t,tm,r,shot,lives,bombs,miss,c,k,flags);
        last=t;if(++count==120)return 0;
        usleep(1500);
    }
    return 3;
}
