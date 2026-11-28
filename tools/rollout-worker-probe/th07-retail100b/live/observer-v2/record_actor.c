/* TH07 1.00b original record-mode Normal. Emit a bracketed game-camera observation, then accept
   the four-head actor's packed key request. Native words and XY are audit-only. */
#define main input_sampler_main
#include "native_input.c"
#undef main
#include <dlfcn.h>
#include <stdlib.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <errno.h>
#include "archive.h"
#include <string.h>
static pid_t game_pid=-1;
static int pausing=0;
static void resume_game(void){if(pausing && game_pid>0){kill(game_pid,SIGCONT);pausing=0;}}
static void on_signal(int sig){if(game_pid>0)kill(game_pid,SIGCONT);_exit(128+sig);}
static int freeze_game(void){
    if (kill(game_pid,SIGSTOP))return 0;
    pausing=1;
    char name[80];snprintf(name,sizeof(name),"/proc/%d/status",game_pid);
    long long wait_limit=mono_ns()+180000000;
    while(mono_ns()<wait_limit){
        char line[100]; FILE *f=fopen(name,"r");if(!f)break;
        int stopped=0;
        while(fgets(line,sizeof(line),f))if(!strncmp(line,"State:",6)){stopped=strchr(line,84)!=NULL;break;}
        fclose(f); if(stopped)return 1;usleep(100);
    }
    resume_game();return 0;
}
static void protect_parent(int fd){
    char path[128],link[128];snprintf(path,sizeof(path),"/proc/self/fd/%d",fd);
    ssize_t n=readlink(path,link,sizeof(link)-1);if(n<=0)exit(19);link[n]=0;
    if(strncmp(link,"/proc/",6))exit(19);
    pid_t pid=0;size_t j=6;while(link[j]>=48&&link[j]<=57){pid=10*pid+(link[j++]-48);}if(pid<1||strcmp(link+j,"/mem"))exit(19);
    game_pid=pid;atexit(resume_game);
    signal(SIGTERM,on_signal);signal(SIGPIPE,on_signal);signal(SIGINT,on_signal);
    /* If this helper vanishes mid-inference, a separate idle process resumes the game.
       It also releases an unresponsive controller after a bounded freeze. */
    pid_t owner=getpid(),watch=fork();if(watch<0)exit(19);
    if(watch==0){
        close(STDIN_FILENO);close(STDOUT_FILENO);
        char name[80];snprintf(name,sizeof(name),"/proc/%d/status",game_pid);
        unsigned stopped=0;
        while(getppid()==owner){
            FILE *f=fopen(name,"r");if(!f)_exit(0);
            char line[128];int is_stopped=0;
            while(fgets(line,sizeof(line),f))if(!strncmp(line,"State:",6)){is_stopped=strchr(line,84)!=NULL;break;}
            fclose(f);stopped=is_stopped?stopped+1:0;
            if(stopped>=16){kill(game_pid,SIGCONT);kill(owner,SIGTERM);_exit(22);}
            usleep(50000);
        }
        kill(game_pid,SIGCONT);_exit(0);
    }
}
typedef struct {int w,h,xoff,fmt;char *data;int order,unit,bitorder,pad,depth,stride,bpp;unsigned long rm,gm,bm;} XI;
typedef struct {uint32_t kind,seq,tick,word,spares,bombs;uint64_t ns;} Packet;
_Static_assert(sizeof(Packet)==32,"packet");
static void *display;
static unsigned long window;
static XI* (*snap)(void*,unsigned long,int,int,unsigned,unsigned,unsigned long,int);
static int (*destroy_image)(XI*);
static int (*key)(void*,unsigned int,int,unsigned long);
static int (*sync_)(void*,int);
static unsigned int codes[7];
static const unsigned bits[7]={16,32,64,128,4,1,2};
static unsigned oldmask;
static void apply(unsigned next) {
    for(int i=0;i<7;i++) if((oldmask^next)&bits[i])key(display,codes[i],!!(next&bits[i]),0);
    sync_(display,0);oldmask=next;
}
static unsigned char reference[192*28], gray[112*96];
static int terminal(XI *im) {
    int n=0,b=0;
    for(int y=0;y<28;y++)for(int x=0;x<192;x++){
        unsigned char *s=(unsigned char*)im->data+(175+y)*im->stride+(114+x)*4;
        int v=(19595*s[2]+38470*s[1]+7471*s[0]+32768)>>16;
        int actual=v>218,ref=reference[192*y+x];
        n+=actual&&ref;b+=actual||ref;
    }
    return b && n*100 > 72*b;
}
static unsigned stock(XI *im,int life){
    int y=life?103:120,n=0;
    for(int x=504;x<=616;x+=16){
        int pixels=0;
        for(int j=-6;j<=6;j++)for(int i=-6;i<=6;i++){
            unsigned char *s=(unsigned char*)im->data+(y+j)*im->stride+(x+i)*4;
            pixels+=life?(s[2]>180&&s[2]-s[0]>30):(s[0]>130&&s[0]-s[2]>35);
        }
        if(pixels<24)break;
        n++;
    }
    return n;
}
static void reduce(XI *im){
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
        int sum=0;
        for(int v=0;v<4;v++){int row=0; unsigned char *s=(unsigned char*)im->data+(16+y*4+v)*im->stride+(32+x*4)*4;
            for(int u=0;u<4;u++,s+=4)row+=(19595*s[2]+38470*s[1]+7471*s[0]+32768)>>16;
            sum+=(row+2)/4;}
        gray[y*96+x]=(sum+2)/4;
    }
}

typedef struct {uint32_t manager,tick,stage,stageclock,flags,pointer,block,rank,score; uint16_t word,previous,record,previous_record; uint8_t retry,continues,shot; float lives,bombs,misses,spent;} Native;
static Native current;
static uint64_t read_worst_ns=0;
static uint32_t word_at(int fd,uint32_t addr){uint32_t u=0;if(pread(fd,&u,4,addr)!=4)return 0;return u;}
/* Read immutable local views of contiguous retail structures. Repeat the identity,
   clock, cursor and both key slots in reverse before publishing any sample. */
static uint64_t latency_bins[10001], polls, rejected, max_tick, max_at;
static int measuring;
static uint32_t u32(const unsigned char *p){uint32_t v;memcpy(&v,p,4);return v;}
static uint16_t u16(const unsigned char *p){uint16_t v;memcpy(&v,p,2);return v;}
static float number(const unsigned char *p){float v;memcpy(&v,p,4);return v;}
static int read_bytes(int fd,void *dst,unsigned n,uint32_t addr){return pread(fd,dst,n,addr)==n;}
static int snapshot(int fd,Native *out){
    unsigned char keys[20], keys2[20], manager[0x88], manager2[0x88], state[0x220], state2[0x220];
    unsigned char rank[12], inventory[0x70], slots[6], slots2[6];
    Native s={0};
    if(!read_bytes(fd,keys,sizeof(keys),0x4b9e48)||!(s.manager=u32(keys)))return 0;
    if(!read_bytes(fd,manager,sizeof(manager),s.manager)||u32(manager+0x44))return 0;
    if(!read_bytes(fd,state,sizeof(state),0x62f640)||!read_bytes(fd,rank,sizeof(rank),0x626278))return 0;
    s.stage=u32(state+0x21c);s.stageclock=u32(state+0x218);s.flags=u32(state+8);
    if(s.stage<1||s.stage>6||(s.flags&11))return 0;
    s.tick=u32(manager);s.pointer=u32(manager+0x84);uint32_t data=u32(manager+4);
    if(!data||!u32(rank)||!read_bytes(fd,&s.block,4,data+28+4*(s.stage-1)))return 0;
    if(!s.block||s.tick>100000||s.pointer!=s.block+44+4*s.tick||s.stageclock!=s.tick)return 0;
    if(!read_bytes(fd,slots,6,s.pointer-4)||!read_bytes(fd,inventory,sizeof(inventory),u32(rank)))return 0;
    s.word=u16(keys+8);s.previous=u16(keys+16);s.record=u16(slots+4);s.previous_record=u16(slots);
    if(s.word!=s.record||(s.tick>0&&s.previous!=s.previous_record))return 0;
    s.retry=state[13];s.shot=state[7];s.rank=u32(rank+8);s.continues=inventory[0x20];
    s.misses=number(inventory+0x50);s.lives=number(inventory+0x5c);s.bombs=number(inventory+0x68);
    s.spent=number(inventory+0x6c);s.score=u32(inventory+4);
    if(!read_bytes(fd,slots2,6,s.pointer-4)||!read_bytes(fd,state2,sizeof(state2),0x62f640)||
       !read_bytes(fd,manager2,sizeof(manager2),s.manager)||!read_bytes(fd,keys2,sizeof(keys2),0x4b9e48))return 0;
    if(s.manager!=u32(keys2)||s.word!=u16(keys2+8)||s.previous!=u16(keys2+16)||
       u32(manager2)!=s.tick||u32(manager2+4)!=data||u32(manager2+0x84)!=s.pointer||
       u32(state2+0x21c)!=s.stage||u32(state2+0x218)!=s.stageclock||u32(state2+8)!=s.flags||
       state2[13]!=s.retry||u16(slots2)!=s.previous_record||u16(slots2+4)!=s.record)return 0;
    *out=s;return 1;
}
static int gather(int fd,uint32_t *tick,uint32_t *word,uint32_t *previous){
    uint64_t start=mono_ns();Native s;int ok=snapshot(fd,&s);uint64_t ended=mono_ns(),elapsed=ended-start;
    if(ok&&(s.flags&4))measuring=1;
    if(measuring){
        polls++;if(!ok)rejected++;
        uint64_t bin=(elapsed+9999)/10000;latency_bins[bin<10000?bin:10000]++;
        if(elapsed>read_worst_ns){read_worst_ns=elapsed;max_tick=ok?s.tick:current.tick;max_at=ended;}
    }
    if(!ok)return 0;
    current=s;*tick=s.tick;*word=s.word;*previous=s.previous;return 1;
}
static void log_native(long long now){
    Native s=current;
    append_event(0,"%lld,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%.0f,%.0f,%.0f,%.0f,%u,%u,%u,%u\n",
        now,s.tick,s.word,s.previous,s.stage,s.stageclock,s.flags,s.retry,s.continues,s.rank,s.shot,
        s.lives,s.bombs,s.misses,s.spent,s.score,s.record,s.previous_record,(s.pointer-s.block-44)/4);
}
static void save_timing(void){
    uint64_t c=0,target=(polls*999+999)/1000;unsigned i;
    for(i=0;i<10000;i++){c+=latency_bins[i];if(c>=target)break;}
    FILE *f=fopen("/capture/observer-timing.txt","wx");if(!f)exit(66);
    fprintf(f,"max_synchronous_native_read_ns=%llu\n",(unsigned long long)read_worst_ns);if(fclose(f))exit(66);
    f=fopen("/capture/observer-distribution.csv","wx");if(!f)exit(66);
    fprintf(f,"polls,rejected,p999_upper_ns,max_ns,max_end_ns,max_tick\n%llu,%llu,%llu,%llu,%llu,%llu\n",
        (unsigned long long)polls,(unsigned long long)rejected,(unsigned long long)i*10000,
        (unsigned long long)read_worst_ns,(unsigned long long)max_at,(unsigned long long)max_tick);
    if(fclose(f))exit(66);
}
static void dump_stage_buffers(int fd,Native *s){
    uint32_t data=word_at(fd,s->manager+4);
    for(unsigned stage=1;stage<=6;stage++){
        uint32_t start=word_at(fd,data+28+4*(stage-1)),end=word_at(fd,s->manager+0x88+4*(stage-1));
        if(!start||end<start+44||end-start>0x70800)continue;
        unsigned n=end-start;unsigned char*bytes=malloc(n);if(!bytes||pread(fd,bytes,n,start)!=n)exit(40);
        char path[80];snprintf(path,sizeof(path),"/capture/record-stage%u.bin",stage);
        FILE*f=fopen(path,"wb");if(!f||fwrite(bytes,1,n,f)!=n||fclose(f))exit(40);free(bytes);
    }
}
int main(void){
    int fd=open_game();if(fd<0)return 2;
    protect_parent(fd);
    void *xl=dlopen("libX11.so.6",RTLD_NOW),*xt=dlopen("libXtst.so.6",RTLD_NOW);if(!xl||!xt)return 3;
    void *(*open_display)(char*)=dlsym(xl,"XOpenDisplay");
    unsigned long (*root_x)(void*)=dlsym(xl,"XDefaultRootWindow");
    unsigned (*keycode)(void*,unsigned long)=dlsym(xl,"XKeysymToKeycode");
    snap=dlsym(xl,"XGetImage");destroy_image=dlsym(xl,"XDestroyImage");key=dlsym(xt,"XTestFakeKeyEvent");sync_=dlsym(xl,"XSync");
    if(!open_display||!root_x||!keycode||!snap||!destroy_image||!key||!sync_)return 4;
    display=open_display(NULL);if(!display)return 5;window=root_x(display);
    const unsigned long symbols[7]={0xff52,0xff54,0xff51,0xff53,0xffe1,0x7a,0x78};
    for(int i=0;i<7;i++){codes[i]=keycode(display,symbols[i]);if(!codes[i])return 6;}
    archive_open();
    uint32_t last=UINT32_MAX,lastword=UINT32_MAX,decision=UINT32_MAX,seq=0;
    long long last_capture=0;int done=0;
    while(!done){
        /* Retail Supervisor::curState 9 is Ending. It is a distinct stock
           route after Stage 6. Keep the final native state and release inputs. */
        if(current.stage==6 && current.continues==0 && word_at(fd,0x575aa8)==9){
            apply(0);
            FILE*ending=fopen("/capture/native-ending-state.txt","wx");
            if(!ending)return 41;
            fprintf(ending,"stage=6 supervisor_state=9 last_tick=%u continues=%u lives=%.0f bombs=%.0f\n",current.tick,current.continues,current.lives,current.bombs);
            fclose(ending);
            dump_stage_buffers(fd,&current);
            Packet packet={3,seq,current.tick,current.word,(unsigned)current.lives,(unsigned)current.bombs,(uint64_t)mono_ns()};
            if(fwrite(&packet,1,sizeof(packet),stdout)!=sizeof(packet))return 42;
            fflush(stdout);done=1;continue;
        }
        uint32_t t,k,kp;
        int rd=gather(fd,&t,&k,&kp);if(rd<0)return 9;if(!rd){usleep(350);continue;}
        long long now=mono_ns();
        if(t!=last||k!=lastword||current.retry){
            log_native(now);last=t;lastword=k;
        }
        int due=t>0&&t%4==0&&t!=decision&&(current.flags&4)&&!current.retry;
        if(!due && !current.retry && now-last_capture<220000000){usleep(450);continue;}
        long long freeze_ns=0;
        if(due){
            /* The stock retail recorder has committed both masks at this clock. */
            if(current.pointer-current.block!=44+4*t || current.word!=current.record || current.previous!=current.previous_record) continue;
            freeze_ns=mono_ns();if(!freeze_game()){fprintf(stderr,"unconfirmed suspend at tick %u, resumed and retrying\n",t);continue;}
            uint32_t frozen;if(pread(fd,&frozen,4,current.manager)!=4)return 20;
            if(frozen!=t){resume_game();continue;}
        }
        XI *im=snap(display,window,0,0,640,480,~0UL,2);
        if(!im||im->bpp!=32||im->rm!=0xff0000||im->bm!=255)return 10;
        last_capture=mono_ns();
        int is_terminal=current.retry && current.lives==0 && current.misses>=3;
        uint32_t te,ke,pe;
        if(gather(fd,&te,&ke,&pe)!=1||te!=t||ke!=k){destroy_image(im);resume_game();continue;}
        if(due)log_native(mono_ns());
        if(!due&&!is_terminal){destroy_image(im);resume_game();continue;}
        Packet packet={is_terminal?2:1,seq,t,k,stock(im,1),stock(im,0),(uint64_t)last_capture};
        if(!is_terminal)reduce(im);
        if(fwrite(&packet,1,sizeof(packet),stdout)!=sizeof(packet))return 11;
        if(!is_terminal&&fwrite(gray,1,sizeof(gray),stdout)!=sizeof(gray))return 12;
        fflush(stdout);
        if(is_terminal){apply(0);dump_stage_buffers(fd,&current);destroy_image(im);resume_game();done=1;continue;}
        struct pollfd incoming={.fd=STDIN_FILENO,.events=POLLIN};
        if(poll(&incoming,1,500)!=1){resume_game();return 21;}
        uint32_t mask;
        if(fread(&mask,4,1,stdin)!=1||mask&~247u){apply(0);return 13;}
        apply(mask);
        uint32_t at; if(pread(fd,&at,4,current.manager)!=4)return 14;
        long long applied=mono_ns();
        resume_game(); long long resumed=mono_ns();
        append_event(1,"%u,%u,%u,%u,%lld,%u,%lld,%lld\n",seq,t,at,mask,applied,at,freeze_ns,resumed);
        decision=t;seq++;destroy_image(im);
        if(seq>40000){apply(0);return 15;}
    }
    apply(0);archive_seal();save_timing();return 0;
}
