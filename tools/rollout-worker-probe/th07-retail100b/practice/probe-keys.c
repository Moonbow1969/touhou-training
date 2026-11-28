#define main old_main
#include "native_input.c"
#undef main
static int field(int fd,void *v,size_t n,uint32_t a){return pread(fd,v,n,a)==(ssize_t)n;}
struct record {uint32_t manager,mode,data,stage,tick,clock,flags,block,cursor;uint16_t keys,prev,saved,previous_saved;};
static int read_record(int fd,struct record *r){
    return field(fd,&r->manager,4,0x4b9e48)&&r->manager
        &&field(fd,&r->mode,4,r->manager+0x44)&&field(fd,&r->data,4,r->manager+4)&&r->data
        &&field(fd,&r->stage,4,0x62f85c)&&r->stage>=1&&r->stage<=6
        &&field(fd,&r->tick,4,r->manager)&&field(fd,&r->clock,4,0x62f858)&&field(fd,&r->flags,4,0x62f648)
        &&field(fd,&r->block,4,r->data+28+4*(r->stage-1))&&r->block
        &&field(fd,&r->cursor,4,r->manager+0x84)&&r->cursor>=r->block+44
        &&field(fd,&r->keys,2,0x4b9e50)&&field(fd,&r->prev,2,0x4b9e58)
        &&field(fd,&r->saved,2,r->cursor)&&field(fd,&r->previous_saved,2,r->cursor-4);
}
int main(void){
    int fd=open_game();if(fd<0)return 1;
    long long until=mono_ns()+14000000000LL;
    puts("ns,stage,stage_tick,manager_tick,rank,shot,spares,bombs,misses,continues,keys,flags,record_mode,block,cursor,previous,record_word,record_previous,stable_record,matched");
    unsigned count=0,last=UINT32_MAX,stage_first=0;
    while(mono_ns()<until){
        uint32_t s=0,m=0,g=0,t=0,tm=0,rank=0,t2=0,s2=0,flags=0;
        uint8_t shot=0,c=0;uint16_t k=0;float lives=0,bombs=0,miss=0;
        if(!field(fd,&s,4,0x62f85c)||!field(fd,&m,4,0x4b9e48)||!m||!field(fd,&g,4,0x626278)||!g||
           !field(fd,&t,4,0x62f858)||!field(fd,&tm,4,m)||!field(fd,&rank,4,0x626280)||
           !field(fd,&shot,1,0x62f647)||!field(fd,&flags,4,0x62f648)||
           !field(fd,&lives,4,g+0x5c)||!field(fd,&bombs,4,g+0x68)||!field(fd,&miss,4,g+0x50)||
           !field(fd,&c,1,g+0x20)||!field(fd,&k,2,0x4b9e50)||!field(fd,&t2,4,0x62f858)||!field(fd,&s2,4,0x62f85c)||
           s!=s2||t!=t2||!(flags&4)||s<1||s>6||!t||t!=tm||last==t){usleep(1000);continue;}
        struct record r={0},r2={0};
        int stable=read_record(fd,&r)&&read_record(fd,&r2)&&memcmp(&r,&r2,sizeof r)==0&&r.stage==s&&r.tick==tm&&r.clock==t&&r.keys==k&&r.flags==flags;
        int matched=stable&&r.mode==0&&r.cursor==r.block+44+4*t&&r.keys==r.saved&&r.prev==r.previous_saved;
        if(!count)stage_first=s;
        if(s!=stage_first)return 2;
        printf("%lld,%u,%u,%u,%u,%u,%.0f,%.0f,%.0f,%u,%u,%u,%u,%u,%u,%u,%u,%u,%d,%d\n",mono_ns(),s,t,tm,rank,shot,lives,bombs,miss,c,k,flags,r.mode,r.block,r.cursor,r.prev,r.saved,r.previous_saved,stable,matched);
        last=t;if(++count==120)return 0;
        usleep(1500);
    }
    return 3;
}
