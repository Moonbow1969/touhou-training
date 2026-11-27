/* Engineering-only: stop after 96 original observations and record both physical inputs. */
#define TH12_FOUR_TICK_INPUT
#define main legacy_worker_entry
#include "worker.c"
#undef main
#include <sys/socket.h>
#include <sys/un.h>

static TickState wait_tick(int stage,int at){
    uint64_t begin=now_ns();
    const struct timespec nap={0,350000};
    for(;;){
        TickState s=native_tick();
        if(s.stage==stage&&s.mode==0&&s.manager_stage==stage&&s.tick>=at)return s;
        if(now_ns()-begin>UINT64_C(3000000000))fail("No advancing original stage tick in three seconds.");
        nanosleep(&nap,NULL);
    }
}

int main(int argc,char **argv){
    if(argc!=2||strlen(argv[1])>=sizeof(((struct sockaddr_un*)0)->sun_path))return 2;
    initialize();open_tick_clock();
    Display *d=XOpenDisplay(NULL);if(!d)fail("Original display missing.");
    Window root=XDefaultRootWindow(d);
    const char *names[7]={"Left","Right","Up","Down","Shift_L","z","x"};
    for(int i=0;i<7;i++)codes[i]=XKeysymToKeycode(d,XStringToKeysym(names[i]));
    int fd=socket(AF_UNIX,SOCK_STREAM,0);
    struct sockaddr_un address={.sun_family=AF_UNIX};strcpy(address.sun_path,argv[1]);
    if(connect(fd,(struct sockaddr*)&address,sizeof(address))<0)fail("Model socket missing.");
    FILE *steps=fopen("subwindow-wire.tsv","wx");
    FILE *frames=fopen("subwindow-frames.u8","wbx");if(!steps||!frames)fail("Prior trace found.");
    fputs("group\timage_stage\timage_tick\tfirst_onset_ns\tfirst_apply_tick\tfirst_x11\tfirst_verified\tmid_stage\tmid_observed_tick\tmid_raw\tsecond_onset_ns\tsecond_apply_tick\tsecond_x11\tsecond_verified\tend_stage\tend_tick\tend_raw\tcamera_retries\n",steps);
    unsigned current=0;int prior=-1;
    for(int i=0;i<96;i++){
        TickState a,b;XImage *im;int retry=0;
        for(;;){
            a=wait_tick(1,prior+1);
            if(a.tick%4){a=wait_tick(1,a.tick+4-a.tick%4);if(a.tick%4)continue;}
            im=image_now(d,root);b=native_tick();
            if(b.stage==1&&b.tick==a.tick&&b.mode==0&&b.manager_stage==1)break;
            retry++;XDestroyImage(im);
        }
        uint8_t frame[SMALL_W*SMALL_H];playfield(im,frame);XDestroyImage(im);
        fwrite(frame,sizeof(frame),1,frames);
        size_t sent=0;
        while(sent<sizeof(frame)){
            ssize_t n=send(fd,frame+sent,sizeof(frame)-sent,MSG_NOSIGNAL);
            if(n<=0)fail("Failed to send original observation.");
            sent+=(size_t)n;
        }
        uint8_t sampled[2][4];if(recv(fd,sampled,sizeof(sampled),MSG_WAITALL)!=(ssize_t)sizeof(sampled))fail("Model response incomplete.");
        unsigned keys[2];
        for(int slot=0;slot<2;slot++){
            int command[4];for(int k=0;k<4;k++){command[k]=sampled[slot][k];if(command[k]>(k?1:8))fail("Invalid policy action.");}
            keys[slot]=mask_for(command);
        }
        uint64_t first=now_ns();set_keys(d,current,keys[0]);unsigned k0=read_keys(d);TickState applied0=native_tick();
        TickState mid=wait_tick(1,a.tick+2);
        uint64_t second=now_ns();set_keys(d,keys[0],keys[1]);unsigned k1=read_keys(d);TickState applied1=native_tick();current=keys[1];
        TickState end=wait_tick(1,a.tick+4);prior=a.tick;
        fprintf(steps,"%d\t%d\t%d\t%llu\t%d\t%u\t%u\t%d\t%d\t%d\t%llu\t%d\t%u\t%u\t%d\t%d\t%d\t%d\n",i,a.stage,a.tick,(unsigned long long)first,applied0.tick,keys[0],k0,mid.stage,mid.tick,mid.keys,(unsigned long long)second,applied1.tick,keys[1],k1,end.stage,end.tick,end.keys,retry);
        if(k0!=keys[0]||k1!=keys[1])fail("Measured X keys differ.");
    }
    set_keys(d,current,0);if(read_keys(d)!=0)fail("Failed to release keys.");
    fclose(steps);fclose(frames);close(fd);XCloseDisplay(d);return 0;
}
