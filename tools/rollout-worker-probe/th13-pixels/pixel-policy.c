#include "x11_pixels.h"
#include <math.h>
#include <string.h>

/* PCG32. The native game RNG remains uncontrolled and separate from action sampling. */
static uint64_t rng;
static uint32_t random32(void) {
    uint64_t old = rng;
    rng = old * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    uint32_t x = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    unsigned rot = (unsigned)(old >> 59u);
    return (x >> rot) | (x << ((-rot) & 31u));
}
int main(int argc, char **argv) {
    if(argc!=5) return 1;
    FILE *model=fopen(argv[1],"rb");
    double w[3][85];
    if(!model || fread(w,sizeof(w),1,model)!=1 || fgetc(model)!=EOF) return 2;
    fclose(model);
    rng=strtoull(argv[2],NULL,10);
    random32();
    long seconds=strtol(argv[3],NULL,10), hold=strtol(argv[4],NULL,10);
    long max_seconds=getenv("TH13_STORY_FOLLOWUP") ? 720 : 180;
    if(seconds<1 || seconds>max_seconds || hold<80 || hold>900) return 3;
    Display *d=XOpenDisplay(NULL);
    if(!d) return 4;
    unsigned char frame[112][96],prior[112][96];
    FILE *pixels=fopen("pixels.u8","wb"),*clocks=fopen("wall.ns","w"),*behavior=fopen("behavior.tsv","w");
    if(!pixels || !clocks || !behavior) return 5;
    fprintf(behavior,"action\tp_left\tp_neutral\tp_right\tlogp\tonset_ns\n");
    unsigned char left=XKeysymToKeycode(d,0xff51),right=XKeysymToKeycode(d,0xff53),shoot=XKeysymToKeycode(d,0x7a);
    capture(d,pixels,clocks,frame);memcpy(prior,frame,sizeof(frame));
    int previous=1;
    XTestFakeKeyEvent(d,shoot,1,0);
    const int64_t end=now_ns()+seconds*1000000000LL;
    const char *stop_file = getenv("TH13_STOP_FILE");
    while(now_ns()<end && (!stop_file || access(stop_file,F_OK)!=0)){
        double f[85]={1.0};
        for(int y=0;y<112;y++) for(int x=0;x<96;x++){
            int k=(y/16)*6+x/16;
            f[1+k]+=frame[y][x]/65280.0;
            f[43+k]+=abs((int)frame[y][x]-(int)prior[y][x])/65280.0;
        }
        double logits[3]={0},probs[3],total=0;
        for(int a=0;a<3;a++) for(int j=0;j<85;j++) logits[a]+=w[a][j]*f[j];
        double max_logit=fmax(logits[0],fmax(logits[1],logits[2]));
        for(int a=0;a<3;a++){probs[a]=exp(logits[a]-max_logit);total+=probs[a];}
        for(int a=0;a<3;a++) probs[a]/=total;
        double u=((double)random32()+0.5)/4294967296.0;
        int a=(u<probs[0])?0:(u<probs[0]+probs[1])?1:2;
        if(a!=previous){
            if(previous!=1) XTestFakeKeyEvent(d,previous==0?left:right,0,0);
            if(a!=1) XTestFakeKeyEvent(d,a==0?left:right,1,0);
        }
        previous=a;XSync(d,0);
        int64_t ts=now_ns();
        fprintf(behavior,"%d\t%.12g\t%.12g\t%.12g\t%.12g\t%lld\n",a,probs[0],probs[1],probs[2],log(probs[a]),(long long)ts);
        long remain=(long)((end-ts)/1000000);
        hold_ms(remain>0 ? (hold<remain?hold:remain) : 1);
        memcpy(prior,frame,sizeof(frame));capture(d,pixels,clocks,frame);
    }
    XTestFakeKeyEvent(d,left,0,0);XTestFakeKeyEvent(d,right,0,0);XTestFakeKeyEvent(d,shoot,0,0);XSync(d,0);
    fclose(pixels);fclose(clocks);fclose(behavior);XCloseDisplay(d);
    return 0;
}
