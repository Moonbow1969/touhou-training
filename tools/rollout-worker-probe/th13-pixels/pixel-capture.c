/* Private Xvfb screen and ordinary XTest keyboard collection. No game process reads. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

typedef struct _XDisplay Display;
typedef struct _XImage XImage;
/* Public Xlib/XTest ABI. The worker runtime includes their versioned shared libraries. */
extern Display *XOpenDisplay(const char *);
extern unsigned long XDefaultRootWindow(Display *);
extern XImage *XGetImage(Display *, unsigned long, int, int, unsigned int, unsigned int, unsigned long, int);
extern unsigned long XGetPixel(XImage *, int, int);
extern int XDestroyImage(XImage *);
extern unsigned char XKeysymToKeycode(Display *, unsigned long);
extern int XTestFakeKeyEvent(Display *, unsigned int, int, unsigned long);
extern int XSync(Display *, int);
extern int XCloseDisplay(Display *);

static int64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) exit(2);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static void hold_ms(long ms) {
    struct timespec ts = {ms / 1000, ms % 1000 * 1000000};
    while (nanosleep(&ts, &ts) && (ts.tv_sec || ts.tv_nsec)) {}
}
static void capture(Display *d, FILE *pixels, FILE *clocklog) {
    XImage *im = XGetImage(d, XDefaultRootWindow(d), 35, 38, 384, 448, ~0UL, 2);
    if (!im) exit(3);
    fprintf(clocklog, "%lld\n", (long long)now_ns());
    unsigned char frame[112][96];
    for (int y=0; y<112; y++) {
        for (int x=0; x<96; x++) {
            unsigned sum=0;
            for (int j=0;j<4;j++) {
                for (int k=0;k<4;k++) {
                    unsigned long v=XGetPixel(im,x*4+k,y*4+j);
                    sum += 77*((v>>16)&255) + 150*((v>>8)&255) + 29*(v&255);
                }
            }
            frame[y][x]=(unsigned char)((sum+2048)/4096);
        }
    }
    XDestroyImage(im);
    if(fwrite(frame,sizeof(frame),1,pixels)!=1) exit(4);
}
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
    capture(d,pixels,clocklog);
    while(scanf("%d %d %d %d",&dx,&dy,&shoot,&focus)==4){
        int next[]={dx<0,dx>0,dy<0,dy>0,shoot,focus};
        for(int i=0;i<6;i++) {
            if(next[i]!=last[i]) XTestFakeKeyEvent(d,keys[i],next[i],0);
            last[i]=next[i];
        }
        XSync(d,0);
        fprintf(onsets,"%lld\n",(long long)now_ns());
        hold_ms(ms);
        capture(d,pixels,clocklog);
    }
    for(int i=0;i<6;i++) if(last[i]) XTestFakeKeyEvent(d,keys[i],0,0);
    XSync(d,0);XCloseDisplay(d);fclose(pixels);fclose(clocklog);fclose(onsets);
    return 0;
}
