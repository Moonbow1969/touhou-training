/* Capture the stock game's Xvfb client and verify the physical seven-key state. */
#include "x11_pixels.h"
extern int XQueryKeymap(Display *, char[32]);

#include "gate.h"
#ifdef TH08_SYNCHRONOUS_ENGINEERING
#include "sync_gate.h"
#endif
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
    fprintf(gate_log,"index stage camera_target camera_observed camera_ns apply_target apply_observed x_ack_ns x11_mask command_received_ns gate_ready_ns\n");
    int index=0,anchor=-1,stage_index=0;
    unsigned anchor_stage=0;
#ifdef TH08_SYNCHRONOUS_ENGINEERING
    if(!bind_sync_game())return 31;
    int failed_stops=0;
#endif
    for(;;) {
        int camera_target=anchor<0?-1:anchor+4*stage_index-2;
#ifdef TH08_SYNCHRONOUS_ENGINEERING
        camera_target=anchor<0?-1:anchor+4*stage_index;
#endif
        int camera_tick=camera_target<0?poll_tick():wait_tick(camera_target);
        if(anchor_stage!=observed_stage || camera_tick<0){anchor=-1;stage_index=0;camera_target=-1;}
        unsigned camera_stage=observed_stage;
#ifdef TH08_SYNCHRONOUS_ENGINEERING
        int latched=camera_tick>0 && freeze_at(camera_stage,camera_tick);
        if(camera_tick>0&&!latched){
            report_quiet_failure(32,camera_stage,camera_tick);
            if(++failed_stops==1)fprintf(stderr,"Native freeze awaiting live heartbeat at stage %u tick %d\n",camera_stage,camera_tick);
            if(failed_stops>=8)return 32;
            continue;
        }
        failed_stops=0;
#endif
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
#ifdef TH08_SYNCHRONOUS_ENGINEERING
        if(latched){struct pollfd command={.fd=STDIN_FILENO,.events=POLLIN};
            if(poll(&command,1,500)!=1){report_quiet_failure(33,camera_stage,camera_tick);sync_resume();return 33;}}
#endif
        if(!fgets(input,sizeof(input),stdin))break;
        int64_t received_ns=now_ns();
#ifdef TH08_SYNCHRONOUS_ENGINEERING
        next=(unsigned)sync_decimal(input);
#else
        next=(unsigned)atoi(input);
#endif
        if(next==128 || next==129){
#ifdef TH08_SYNCHRONOUS_ENGINEERING
            sync_resume();
#endif
            /* A capped engineering stop can hold its last scheduled action to completion. */
            if(next==129 && anchor>=0)wait_tick(anchor+4*stage_index);
            break;
        }
        int is_policy=(next&256)!=0;
        next &= 127;
        int apply_target=anchor<0?-1:anchor+4*stage_index;
#ifdef TH08_SYNCHRONOUS_ENGINEERING
        int apply_tick=-1;
        if(is_policy&&(!latched||!frozen_matches(camera_stage,camera_tick))){report_quiet_failure(34,camera_stage,camera_tick);sync_resume();return 34;}
        if(is_policy){apply_tick=camera_tick;apply_target=camera_tick;}
#else
        int apply_tick=is_policy?(apply_target<0?poll_tick():wait_tick(apply_target)):-1;
#endif
        int skipped=is_policy && (apply_tick<0 || camera_stage!=observed_stage);
        if(skipped){is_policy=0;next=0;anchor=-1;stage_index=0;}
        if(is_policy && anchor<0){anchor=apply_tick;apply_target=anchor;anchor_stage=observed_stage;}
        int64_t gate_ready_ns=now_ns();
        for(int k=0;k<7;k++)if(((next^current)>>k)&1u)XTestFakeKeyEvent(d,keys[k],(next>>k)&1u,0);
        XSync(d,0);
        char keymap[32];XQueryKeymap(d,keymap);
        unsigned observed=0;
        for(int k=0;k<7;k++)if(((unsigned char)keymap[keys[k]/8]>>(keys[k]%8))&1u)observed|=1u<<k;
#ifdef TH08_SYNCHRONOUS_ENGINEERING
        if(latched){
            int64_t ack_ns=now_ns();
#ifdef TH08_QUIET_READS
            quiet_release();
            if(!quiet_tick_proof(camera_stage,camera_tick,&post_read_serial)){report_quiet_failure(36,camera_stage,camera_tick);sync_resume();return 36;}
            if(is_policy){fprintf(quiet_log,"%d %u %d %u %u %lld\n",index,camera_stage,camera_tick,frozen_proven_serial,post_read_serial,(long long)last_quiet_acquired_ns);fflush(quiet_log);}
#endif
            if(!frozen_matches(camera_stage,camera_tick)){report_quiet_failure(35,camera_stage,camera_tick);sync_resume();return 35;}
            sync_resume();int64_t resumed=now_ns();
            if(is_policy){
                fprintf(freeze_receipt,"%d %d %u %d %lld %lld %lld %lld %lld %lld %lld %u %u\n",index,(int)sync_pid,camera_stage,camera_tick,
                        (long long)freeze_start_ns,(long long)freeze_stop_ns,(long long)freeze_proof_ns,(long long)time,(long long)received_ns,
                        (long long)ack_ns,(long long)resumed,frozen_first_serial,frozen_proven_serial);
                fflush(freeze_receipt);
            }
        }
#endif
        printf("%s %u %lld\n",skipped?"SKIP":"KEY",observed,(long long)now_ns());fflush(stdout);
        if(observed!=next)return 5;
        current=next;
        if(is_policy){
            fprintf(gate_log,"%d %u %d %d %lld %d %d %lld %u %lld %lld\n",index,observed_stage,camera_target,camera_tick,(long long)time,
                    apply_target,apply_tick,(long long)now_ns(),observed,(long long)received_ns,(long long)gate_ready_ns);
            fflush(gate_log);index++;stage_index++;
        } else hold_ms(60);
    }
    for(int k=0;k<7;k++)XTestFakeKeyEvent(d,keys[k],0,0);
    XSync(d,0);XCloseDisplay(d);
    return 0;
}
