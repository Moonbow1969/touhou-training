/* Full-action replay/PPO actor for one free-running official TH10 Normal credit. */
#include "x11_pixels.h"
#include "full_actor.h"
#include <inttypes.h>
#include <string.h>
#include <signal.h>

static Display *display;
static unsigned char keys[7];
static unsigned latched;
static void set_mask(unsigned mask) {
    for(unsigned i=0;i<7;i++)if(((mask^latched)>>i)&1u)XTestFakeKeyEvent(display,keys[i],(mask>>i)&1u,0);
    latched=mask; XSync(display,0);
}
static void release_keys(void){if(display){set_mask(0);XCloseDisplay(display);display=NULL;}}
static int64_t realtime_ms(void){struct timespec t;clock_gettime(CLOCK_REALTIME,&t);return (int64_t)t.tv_sec*1000+t.tv_nsec/1000000;}
static unsigned key_mask(const int a[4]) {
    int dx=a[0]%3-1, dy=a[0]/3-1;
    return (dx<0?1u:dx>0?2u:0u)|(dy<0?4u:dy>0?8u:0u)|(a[1]?16u:0u)|(a[2]?32u:0u)|(a[3]?64u:0u);
}
int main(int argc,char **argv) {
    if(argc!=4)return 2;
    uint64_t seed=strtoull(argv[2],NULL,10);long tick=strtol(argv[3],NULL,10);
    if(tick<70 || tick>350 || actor_init(argv[1],seed)!=0)return 3;
    display=XOpenDisplay(NULL);if(!display)return 4;
    const unsigned long syms[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78};
    for(unsigned i=0;i<7;i++)keys[i]=XKeysymToKeycode(display,syms[i]);
    atexit(release_keys);
    FILE *pixel=fopen("pixels.u8","wb"),*clock=fopen("frame.ns","w"),*tab=fopen("action.csv","w"),*feat=fopen("features.f32","wb");
    if(!pixel||!clock||!tab||!feat)return 5;
    fprintf(tab,"utc_ms,mono_ns,move,focus,shoot,bomb,mask,lp_move,lp_focus,lp_shoot,lp_bomb,value\n");
    unsigned char frame[112][96];
    capture(display,pixel,clock,frame);
    int64_t end=now_ns()+INT64_C(45)*60*1000000000;
    while(access("native-stop",F_OK)!=0 && now_ns()<end) {
        int a[4];actor_step(&frame[0][0],a);
        if(a[0]<0||a[0]>8||a[1]<0||a[1]>1||a[2]<0||a[2]>1||a[3]<0||a[3]>1)return 6;
        unsigned mask=key_mask(a);
        set_mask(mask);
        fprintf(tab,"%" PRId64 ",%" PRId64 ",%d,%d,%d,%d,%u,%.9g,%.9g,%.9g,%.9g,%.9g\n",realtime_ms(),now_ns(),a[0],a[1],a[2],a[3],mask,
                (double)actor_logp[0],(double)actor_logp[1],(double)actor_logp[2],(double)actor_logp[3],(double)actor_value);
        const size_t feature_dim=sizeof(actor_features)/sizeof(actor_features[0]);
        if(fwrite(actor_features,sizeof(float),feature_dim,feat)!=feature_dim)return 7;
        hold_ms(tick);capture(display,pixel,clock,frame);
    }
    release_keys();
    fclose(feat);fclose(tab);fclose(pixel);fclose(clock);
    return 0;
}
