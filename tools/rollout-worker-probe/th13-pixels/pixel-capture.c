#include "x11_pixels.h"

int main(int argc,char **argv) {
    if(argc!=2) return 1;
    long ms=strtol(argv[1],NULL,10);
    if(ms<25 || ms>250) return 1;
    Display *d=XOpenDisplay(NULL);
    if(!d) return 2;
    FILE *pixels=fopen("pixels.u8","wb"), *clocklog=fopen("wall.ns","w"), *onsets=fopen("onset.ns","w");
    if(!pixels || !clocklog || !onsets) return 3;
    const unsigned long syms[]={0xff51,0xff53,0xff52,0xff54,0x7a,0xffe1};
    unsigned char keys[6];
    int last[6]={0},dx,dy,shoot,focus;
    for(int i=0;i<6;i++) keys[i]=XKeysymToKeycode(d,syms[i]);
    unsigned char frame[112][96];
    capture(d,pixels,clocklog,frame);
    while(scanf("%d %d %d %d",&dx,&dy,&shoot,&focus)==4){
        int next[]={dx<0,dx>0,dy<0,dy>0,shoot,focus};
        for(int i=0;i<6;i++) {
            if(next[i]!=last[i]) XTestFakeKeyEvent(d,keys[i],next[i],0);
            last[i]=next[i];
        }
        XSync(d,0);
        fprintf(onsets,"%lld\n",(long long)now_ns());
        hold_ms(ms);
        capture(d,pixels,clocklog,frame);
    }
    for(int i=0;i<6;i++) if(last[i]) XTestFakeKeyEvent(d,keys[i],0,0);
    XSync(d,0);XCloseDisplay(d);fclose(pixels);fclose(clocklog);fclose(onsets);
    return 0;
}
