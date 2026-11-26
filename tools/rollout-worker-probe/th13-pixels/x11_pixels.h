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
#ifndef PLAYFIELD_LEFT
#define PLAYFIELD_LEFT 35
#define PLAYFIELD_TOP 38
#endif
static void capture(Display *d, FILE *pixels, FILE *clocklog, unsigned char frame[112][96]) {
    XImage *im = XGetImage(d, XDefaultRootWindow(d), PLAYFIELD_LEFT, PLAYFIELD_TOP, 384, 448, ~0UL, 2);
    if (!im) exit(3);
    fprintf(clocklog, "%lld\n", (long long)now_ns());
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
    if(fwrite(frame,112*96,1,pixels)!=1) exit(4);
}
