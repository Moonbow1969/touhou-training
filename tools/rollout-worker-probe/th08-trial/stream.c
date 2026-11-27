/* Capture the stock game's Xvfb client and verify the physical seven-key state. */
#include "../th13-pixels/x11_pixels.h"
extern int XQueryKeymap(Display *, char[32]);

int main(void) {
    const unsigned long syms[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78};
    Display *d=XOpenDisplay(NULL);
    if(!d)return 2;
    unsigned char keys[7];
    unsigned current=0;
    for(int k=0;k<7;k++)keys[k]=XKeysymToKeycode(d,syms[k]);
    static unsigned char rgb[480][640][3];
    for(;;) {
        XImage *im=XGetImage(d,XDefaultRootWindow(d),0,0,640,480,~0UL,2);
        if(!im)return 3;
        int64_t time=now_ns();
        for(int y=0;y<480;y++)for(int x=0;x<640;x++){
            unsigned long v=XGetPixel(im,x,y);
            rgb[y][x][0]=(v>>16)&255;rgb[y][x][1]=(v>>8)&255;rgb[y][x][2]=v&255;
        }
        XDestroyImage(im);
        if(printf("RGB %lld\n",(long long)time)<0 || fwrite(rgb,1,sizeof(rgb),stdout)!=sizeof(rgb) || fflush(stdout))return 4;
        unsigned next;
        if(scanf("%u",&next)!=1 || next>127)break;
        for(int k=0;k<7;k++)if(((next^current)>>k)&1u)XTestFakeKeyEvent(d,keys[k],(next>>k)&1u,0);
        XSync(d,0);
        char keymap[32];XQueryKeymap(d,keymap);
        unsigned observed=0;
        for(int k=0;k<7;k++)if(((unsigned char)keymap[keys[k]/8]>>(keys[k]%8))&1u)observed|=1u<<k;
        printf("KEY %u %lld\n",observed,(long long)now_ns());fflush(stdout);
        if(observed!=next)return 5;
        current=next;
        hold_ms(85);
    }
    for(int k=0;k<7;k++)XTestFakeKeyEvent(d,keys[k],0,0);
    XSync(d,0);XCloseDisplay(d);
    return 0;
}
