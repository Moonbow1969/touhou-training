/* TH07 0.11 original Normal. Emit a bracketed game-camera observation, then accept
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
static int gather(int fd,uint32_t *tick,uint32_t *word,uint32_t *previous){
    uint32_t t2,k2;
    if(pread(fd,tick,4,0x11c9ae8)!=4||pread(fd,word,4,0x11c9d24)!=4||
       pread(fd,previous,4,0x11c9d2c)!=4||pread(fd,&k2,4,0x11c9d24)!=4||
       pread(fd,&t2,4,0x11c9ae8)!=4)return -1;
    return *tick==t2&&*word==k2;
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
    FILE *ref=fopen("/tool/continue-mask.bin","rb");if(!ref||fread(reference,1,sizeof(reference),ref)!=sizeof(reference))return 7;fclose(ref);
    FILE *log=fopen("/capture/native-input.csv","wx"),*requests=fopen("/capture/applied.csv","wx");
    if(!log||!requests)return 8;
    fprintf(log,"ns,tick,word,previous_word\n");fprintf(requests,"sequence,camera_tick,applied_tick,requested_word,apply_ns,frozen_tick,freeze_start_ns,resume_ns\n");
    uint32_t last=UINT32_MAX,lastword=UINT32_MAX,decision=UINT32_MAX,seq=0;
    long long last_capture=0;int done=0;
    while(!done){
        uint32_t t,k,kp;
        int rd=gather(fd,&t,&k,&kp);if(rd<0)return 9;if(!rd)continue;
        long long now=mono_ns();
        if(t!=last||k!=lastword){
            fprintf(log,"%lld,%u,%u,%u\n",now,t,k&65535,kp&65535);last=t;lastword=k;
        }
        int due=t>0&&t%4==0&&t!=decision;
        if(!due && now-last_capture<220000000){usleep(450);continue;}
        long long freeze_ns=0;
        if(due){
            /* Allow this tick to finish its input poll before clamping the next decision. */
            usleep(1500);
            freeze_ns=mono_ns();if(!freeze_game()){fprintf(stderr,"unconfirmed suspend at tick %u, resumed and retrying\n",t);continue;}
            uint32_t frozen;if(pread(fd,&frozen,4,0x11c9ae8)!=4)return 20;
            if(frozen!=t){resume_game();continue;}
        }
        XI *im=snap(display,window,0,0,640,480,~0UL,2);
        if(!im||im->bpp!=32||im->rm!=0xff0000||im->bm!=255)return 10;
        last_capture=mono_ns();
        int is_terminal=terminal(im);
        uint32_t te,ke,pe;
        if(gather(fd,&te,&ke,&pe)!=1||te!=t||ke!=k){destroy_image(im);resume_game();continue;}
        if(!due&&!is_terminal){destroy_image(im);resume_game();continue;}
        Packet packet={is_terminal?2:1,seq,t,k,stock(im,1),stock(im,0),(uint64_t)last_capture};
        if(!is_terminal)reduce(im);
        if(fwrite(&packet,1,sizeof(packet),stdout)!=sizeof(packet))return 11;
        if(!is_terminal&&fwrite(gray,1,sizeof(gray),stdout)!=sizeof(gray))return 12;
        fflush(stdout);
        if(is_terminal){destroy_image(im);resume_game();done=1;continue;}
        struct pollfd incoming={.fd=STDIN_FILENO,.events=POLLIN};
        if(poll(&incoming,1,500)!=1){resume_game();return 21;}
        uint32_t mask;
        if(fread(&mask,4,1,stdin)!=1||mask&~247u){apply(0);return 13;}
        apply(mask);
        uint32_t at; if(pread(fd,&at,4,0x11c9ae8)!=4)return 14;
        long long applied=mono_ns();
        resume_game(); long long resumed=mono_ns();
        fprintf(requests,"%u,%u,%u,%u,%lld,%u,%lld,%lld\n",seq,t,at,mask,applied,at,freeze_ns,resumed);
        if(seq%64==0){fflush(requests);fflush(log);}
        decision=t;seq++;destroy_image(im);
        if(seq>40000){apply(0);return 15;}
    }
    apply(0);if(fclose(log)||fclose(requests))return 16;return 0;
}
