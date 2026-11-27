/* Forward a diagnostic controller's ordinary movement/focus/shoot/bomb requests to X11. */
#include "x11_pixels.h"
int main(void){
    Display *d=XOpenDisplay(NULL);if(!d)return 2;
    const unsigned long syms[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78};
    FILE *pixels=fopen("witness-camera.u8","wb"),*clock=fopen("witness.ns","w"),*utc=fopen("witness-camera.ms","w");
    if(!pixels||!clock||!utc)return 4;
    uint8_t image[112][96];int64_t last_frame=0;
    int key[7];for(int i=0;i<7;i++)key[i]=XKeysymToKeycode(d,syms[i]);
    unsigned prev=0,mask=0;
    while(scanf("%u",&mask)==1){
        if(mask>127)return 3;
        for(int i=0;i<7;i++)if((mask^prev)&(1u<<i))XTestFakeKeyEvent(d,key[i],(mask>>i)&1u,0);
        XSync(d,0);prev=mask;
        if(now_ns()-last_frame>=120000000){
            struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
            fprintf(utc,"%lld\n",(long long)t.tv_sec*1000+t.tv_nsec/1000000);
            capture(d,pixels,clock,image);last_frame=now_ns();
        }
    }
    for(int i=0;i<7;i++)if(prev&(1u<<i))XTestFakeKeyEvent(d,key[i],0,0);
    XSync(d,0);XCloseDisplay(d);fclose(pixels);fclose(clock);fclose(utc);return 0;
}
