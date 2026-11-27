#include "hud.h"
#include "camera_actor.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <stdatomic.h>

typedef struct {
    int32_t stage,start_stage,tick,level_tick,total_tick,character,beast,score,rank,continues,power,lives,life_pieces,bombs,bomb_pieces;
} State;
typedef struct { _Atomic int32_t sequence; volatile State state; } Shared;
static Shared *shared;
static State read_state(void) {
    State state;
    for(;;){
        int seq=atomic_load(&shared->sequence);
        if(!seq || seq%2)continue;
        state=shared->state;
        atomic_thread_fence(memory_order_acquire);
        if(seq==atomic_load(&shared->sequence))return state;
    }
}
static unsigned codes[7];
static unsigned mask(const int a[4]){
    int x=a[0]%3-1,y=a[0]/3-1;
    return ((x<0)<<0)|((x>0)<<1)|((y<0)<<2)|((y>0)<<3)|(a[1]<<4)|(a[2]<<5)|(a[3]<<6);
}
static unsigned apply(Display*d,unsigned old,unsigned next) {
    for(int k=0;k<7;k++)if((old^next)&(1u<<k))XTestFakeKeyEvent(d,codes[k],!!(next&(1u<<k)),0);
    XSync(d,0);char observed[32];XQueryKeymap(d,observed);unsigned result=0;
    for(int k=0;k<7;k++)if((unsigned char)observed[codes[k]/8]&(1u<<(codes[k]%8)))result|=1u<<k;
    return result;
}
static void capture(Display*d,uint8_t *frame) {
    XImage *im=XGetImage(d,XDefaultRootWindow(d),SCENE_X,SCENE_Y,WIDTH,HEIGHT,~0UL,2);
    if(!im)fail("Camera image unavailable");
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
        unsigned sum=0;for(int j=0;j<4;j++)for(int i=0;i<4;i++){
            unsigned long v=XGetPixel(im,4*x+i,4*y+j);sum+=77*((v>>16)&255)+150*((v>>8)&255)+29*(v&255);
        }
        frame[y*96+x]=(uint8_t)((sum+2048)/4096);
    }
    XDestroyImage(im);
}
int main(int argc,char**argv) {
    if(argc!=3)return 2;
    uint64_t seed=strtoull(argv[1],NULL,10);int rank=atoi(argv[2]);
    if(!seed || (rank!=2&&rank!=3))return 3;
    if(actor_init(NULL,seed))fail("Camera weight signature mismatch or service unavailable");
    initialize();Display *d=XOpenDisplay(NULL);if(!d)fail("Private X unavailable");
    const char*names[7]={"Left","Right","Up","Down","Shift_L","z","x"};
    for(int k=0;k<7;k++)codes[k]=XKeysymToKeycode(d,XStringToKeysym(names[k]));
    int fd=open("live.state",O_RDONLY);if(fd<0)fail("Missing witness");
    shared=mmap(NULL,sizeof(*shared),PROT_READ,MAP_SHARED,fd,0);if(shared==MAP_FAILED)fail("Witness mmap failed");
    FILE *rows=fopen("decisions.tsv","wx"), *pixels=fopen("observations.u8","wbx");
    if(!rows||!pixels)fail("Output already exists or invalid");
    fprintf(rows,"action_t\tonset_ns\tobserved_ns\tmove\tfocus\tshoot\tbomb\tkeys\tobserved_keys\tstage\ttick\ttotal_tick\tspares_before\tbombs_before\tafter_stage\tafter_tick\tafter_total_tick\tspares\tbombs\tlogp_move\tlogp_focus\tlogp_shoot\tlogp_bomb\tvalue\n");
    struct timespec nap={0,5000000};uint64_t begin=now_ns();
    State pre;
    do {
        pre=read_state();
        if(pre.stage==1&&pre.rank==rank&&pre.tick>1&&pre.tick<180&&pre.lives==2&&pre.bombs==3&&!pre.continues)break;
        nanosleep(&nap,NULL);if(now_ns()-begin>180000000000ULL)return 8;
    }while(1);
    int i=0,good=0;unsigned current=0;
    for(;;){
        pre=read_state();if(pre.stage<1||pre.rank!=rank||pre.continues||pre.lives<0)break;
        uint8_t frame[112*96];capture(d,frame);int action[4];actor_step(frame,action);if(action[0]<0)break;
        unsigned next=mask(action);uint64_t onset=now_ns();
        unsigned observed=apply(d,current,next);current=next;
        if(observed!=next)break;
        State post;
        do{nanosleep(&nap,NULL);post=read_state();}while(now_ns()-onset<120000000ULL && post.lives>=0);
        uint64_t end=now_ns();
        fwrite(frame,sizeof frame,1,pixels);
        fprintf(rows,"%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%u\t%u\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\n",
                 i,(unsigned long long)onset,(unsigned long long)end,action[0],action[1],action[2],action[3],next,observed,pre.stage,pre.tick,pre.total_tick,pre.lives,pre.bombs,
                 post.stage,post.tick,post.total_tick,post.lives,post.bombs,actor_logp[0],actor_logp[1],actor_logp[2],actor_logp[3],actor_value);
        ++i;
        if(post.continues||post.rank!=rank||post.stage<1)break;
        if(post.lives==-1){good=1;break;}
        if(now_ns()-begin>3600000000000ULL)break;
    }
    apply(d,current,0);fclose(rows);fclose(pixels);return good?0:9;
}
