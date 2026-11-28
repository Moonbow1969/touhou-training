/* Capture the stock game's Xvfb client and verify the physical seven-key state. */
#include "x11_pixels.h"
extern int XQueryKeymap(Display *, char[32]);

#include "gate.h"
int main(void) {
    const unsigned long syms[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78};
    Display *d=XOpenDisplay(NULL);
    if(!d)return 2;
    unsigned char keys[7];
    unsigned current=0;
    for(int k=0;k<7;k++)keys[k]=XKeysymToKeycode(d,syms[k]);
    static unsigned char rgb[480][640][3];
    int fd=open("/capture/current.tick",O_RDONLY);
    if(fd<0)return 12;
    tick_source=mmap(NULL,sizeof(*tick_source),PROT_READ,MAP_SHARED,fd,0);
    if(tick_source==MAP_FAILED)return 12;
    gate_log=fopen("/capture/gates.tsv","wx"); if(!gate_log)return 12;
    fprintf(gate_log,"index stage camera_target camera_observed camera_ns apply_target apply_observed x_ack_ns x11_mask\n");
    int index=0,anchor=-1,stage_index=0;
    unsigned anchor_stage=0;
    for(;;) {
        int camera_target=anchor<0?-1:anchor+4*stage_index-2;
        int camera_tick=camera_target<0?poll_tick():wait_tick(camera_target);
        if(anchor_stage!=observed_stage || camera_tick<0){anchor=-1;stage_index=0;camera_target=-1;}
        unsigned camera_stage=observed_stage;
        XImage *im=XGetImage(d,XDefaultRootWindow(d),0,0,640,480,~0UL,2);
        if(!im)return 3;
        int64_t time=now_ns();
        for(int y=0;y<480;y++)for(int x=0;x<640;x++){
            unsigned long v=XGetPixel(im,x,y);
            rgb[y][x][0]=(v>>16)&255;rgb[y][x][1]=(v>>8)&255;rgb[y][x][2]=v&255;
        }
        XDestroyImage(im);
        if(printf("RGB %lld %u %d\n",(long long)time,camera_stage,camera_tick)<0 || fwrite(rgb,1,sizeof(rgb),stdout)!=sizeof(rgb) || fflush(stdout))return 4;
        unsigned next;
        char input[40];
        if(!fgets(input,sizeof(input),stdin))break;
        next=(unsigned)atoi(input);
        if(next==128 || next==129){
            /* A capped engineering stop can hold its last scheduled action to completion. */
            if(next==129 && anchor>=0)wait_tick(anchor+4*stage_index);
            break;
        }
        int is_policy=(next&256)!=0;
        next &= 127;
        int apply_target=anchor<0?-1:anchor+4*stage_index;
        int apply_tick=is_policy?(apply_target<0?poll_tick():wait_tick(apply_target)):-1;
        if(is_policy && (apply_tick<0 || camera_stage!=observed_stage))return 14;
        if(is_policy && anchor<0){anchor=apply_tick;apply_target=anchor;anchor_stage=observed_stage;}
        for(int k=0;k<7;k++)if(((next^current)>>k)&1u)XTestFakeKeyEvent(d,keys[k],(next>>k)&1u,0);
        XSync(d,0);
        char keymap[32];XQueryKeymap(d,keymap);
        unsigned observed=0;
        for(int k=0;k<7;k++)if(((unsigned char)keymap[keys[k]/8]>>(keys[k]%8))&1u)observed|=1u<<k;
        printf("KEY %u %lld\n",observed,(long long)now_ns());fflush(stdout);
        if(observed!=next)return 5;
        current=next;
        if(is_policy){
            fprintf(gate_log,"%d %u %d %d %lld %d %d %lld %u\n",index,observed_stage,camera_target,camera_tick,(long long)time,
                    apply_target,apply_tick,(long long)now_ns(),observed);fflush(gate_log);index++;stage_index++;
        } else hold_ms(60);
    }
    for(int k=0;k<7;k++)XTestFakeKeyEvent(d,keys[k],0,0);
    XSync(d,0);XCloseDisplay(d);
    return 0;
}
