#include "x11_pixels.h"
#include "camera_actor.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>

static int stock(int fd,int32_t row[12]){
    struct stat info;
    if(fstat(fd,&info) || info.st_size<48)return 0;
    return pread(fd,row,48,(info.st_size/48-1)*48)==48;
}
static void keys(Display *d,const uint8_t *codes,uint8_t *prior,unsigned mask) {
    const int bits[7]={64,128,16,32,8,1,2};
    for(int j=0;j<7;j++){
        int on=(mask&bits[j])!=0;
        if(on!=prior[j])XTestFakeKeyEvent(d,codes[j],on,0);
        prior[j]=on;
    }
    XSync(d,0);
}
int main(int argc,char **argv){
    if(argc!=3)return 1;
    int rank=atoi(argv[2]);
    if(actor_init("",strtoull(argv[1],NULL,10)))return 2;
    Display *d=XOpenDisplay(NULL);if(!d)return 3;
    const unsigned long symbols[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78};
    uint8_t codes[7],prior[7]={0},frame[112][96];
    for(int j=0;j<7;j++)codes[j]=XKeysymToKeycode(d,symbols[j]);
    FILE *pixels=fopen("pixels.u8","wb"),*clock=fopen("camera.ns","w"),*behavior=fopen("behavior.csv","w");
    int fd=open("native.bin",O_RDONLY),active=0,terminal=0;
    if(!pixels||!clock||!behavior||fd<0)return 4;
    int32_t obs[12],after[12];int64_t deadline=now_ns()+1200LL*1000000000LL;
    fprintf(behavior,"step,dir,focus,shoot,bomb,logp_dir,logp_focus,logp_shoot,logp_bomb,value,mask,before_ms,after_ms,observed_mask,score,power,spares,rank,stage,continues\n");
    for(int t=0;now_ns()<deadline;){
        if(!stock(fd,obs)){hold_ms(5);continue;}
        if(!active){
            if(obs[5]!=2||obs[6]!=rank||obs[7]!=1||obs[8]!=0||obs[9]!=20){hold_ms(5);continue;}
            active=1;
        }
        if(obs[5]==-1){terminal=1;break;}
        if(obs[5]<0||obs[5]>9||obs[6]!=rank||obs[8]!=0)break;
        capture(d,pixels,clock,frame);
        int a[4];actor_step((uint8_t*)frame,a);if(a[0]<0)break;
        int x=a[0]%3,y=a[0]/3;
        int mask=(x==0?64:x==2?128:0)|(y==0?16:y==2?32:0)|(a[1]?8:0)|(a[2]?1:0)|(a[3]?2:0);
        keys(d,codes,prior,mask);
        int64_t next=now_ns()+120000000;
        do {
            hold_ms(5);
            if(!stock(fd,after))return 5;
            if(after[5]==-1){terminal=1;keys(d,codes,prior,0);break;}
        } while(now_ns()<next);
        fprintf(behavior,"%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%u,%u,%d,%d,%d,%d,%d,%d,%d\n",
            t++,a[0],a[1],a[2],a[3],actor_logp[0],actor_logp[1],actor_logp[2],actor_logp[3],actor_value,mask,
            (unsigned)obs[0],(unsigned)after[0],after[11]&251,after[1],after[2],after[5],after[6],after[7],after[8]);
        if(terminal)break;
    }
    keys(d,codes,prior,0);close(fd);fclose(behavior);fclose(clock);fclose(pixels);XCloseDisplay(d);
    return !terminal;
}
