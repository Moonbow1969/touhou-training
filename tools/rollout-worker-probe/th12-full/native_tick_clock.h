/* Engineering-only TH12 0.02a game-clock sampling. This telemetry controls camera timing only. */
#include <fcntl.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <unistd.h>
typedef struct {int32_t stage,manager_stage,tick,mode,keys,spares,bombs;} TickState;
typedef struct {_Atomic int32_t seq;volatile TickState v;} TickShared;
static TickShared *tick_shared;
static void open_tick_clock(void){
    int fd=open("engineering-clock.map",O_RDONLY);
    if(fd<0)fail("No original native tick shared mapping.");
    tick_shared=mmap(NULL,sizeof(*tick_shared),PROT_READ,MAP_SHARED,fd,0);
    close(fd);
    if(tick_shared==MAP_FAILED)fail("Cannot map original native tick.");
}
static TickState native_tick(void){
    for(;;){
        int32_t seq=atomic_load(&tick_shared->seq);
        if(seq<=0 || seq%2)continue;
        TickState value=tick_shared->v;
        atomic_thread_fence(memory_order_acquire);
        if(atomic_load(&tick_shared->seq)==seq)return value;
    }
}
/* A photographed terminal is also returned while the gameplay timer is frozen.
   Regular observations are accepted only inside one exact fourth native tick. */
static XImage *tick_camera(Display *d,Window root,TickState prior,TickState *observed,int *rejects,int *terminal_checks){
    const struct timespec nap={0,1000000};
    uint64_t start=now_ns(),last_menu=start;
    *rejects=0;*terminal_checks=0;
    for(;;){
        TickState a=native_tick();
        if(a.mode==0 && a.stage>=1 && a.stage==a.manager_stage && a.tick>=0 && a.tick%4==0 &&
           (a.stage!=prior.stage || a.tick>prior.tick)){
            XImage *image=image_now(d,root);TickState b=native_tick();
            if(b.stage==a.stage && b.manager_stage==a.manager_stage && b.mode==a.mode && b.tick==a.tick){*observed=b;return image;}
            XDestroyImage(image);(*rejects)++;
        }
        uint64_t now=now_ns();
        if(now-last_menu>=UINT64_C(125000000)){
            XImage *image=image_now(d,root);const char *status=screen(image,0);
            (*terminal_checks)++;last_menu=now;
#ifdef TH12_NATIVE_LIVE_SCREEN
            if(!strncmp(status,"pause",5) || !strcmp(status,"confirm") || !strcmp(status,"title")){
                *observed=native_tick();return image;
            }
#endif
            if(!strcmp(status,"gameover") || !strcmp(status,"gameover_retry") || !strcmp(status,"score_ranking")){
                *observed=native_tick();return image;
            }
            XDestroyImage(image);
        }
        if(now-start>UINT64_C(12000000000))fail("Native game clock stalled for twelve seconds with no verified terminal.");
        nanosleep(&nap,NULL);
    }
}
/* Read both native edges without introducing a phase wait into the 120ms control. */
static XImage *wall_camera(Display *display,Window root,TickState *state,int *edge_changed,int *checks){
    TickState before=native_tick();
    XImage *image=image_now(display,root);
    *state=native_tick();
    *edge_changed=(before.tick!=state->tick || before.stage!=state->stage);
    *checks=0;
    return image;
}

