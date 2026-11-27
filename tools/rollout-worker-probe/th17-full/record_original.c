/* Record the private original 640x480 X client. Actor and input run independently. */
#include "hud.h"
#include <unistd.h>
typedef struct {int width,height,xoffset,format;char *data;int byte_order,bitmap_unit,bitmap_bit_order,bitmap_pad,depth,bytes_per_line,bits_per_pixel;} XImagePrefix;
int main(int argc,char**argv) {
    if(argc!=2)return 2;
    initialize();Display *d=XOpenDisplay(NULL);if(!d)return 3;
    uint64_t begin=now_ns(),terminal=0,next=begin;
    for(;;){
        uint64_t t=now_ns();
        if(!terminal&&!access(argv[1],F_OK))terminal=t;
        if(terminal && t-terminal>3000000000ULL)break;
        if(t-begin>900000000000ULL)return 4;
        XImage *image=XGetImage(d,XDefaultRootWindow(d),3,22,640,480,~0UL,2);
        XImagePrefix *p=(XImagePrefix*)image;
        if(!p||p->width!=640||p->height!=480||p->byte_order!=0||p->bits_per_pixel!=32||p->bytes_per_line!=2560)return 5;
        if(fwrite(&t,sizeof t,1,stdout)!=1||fwrite(p->data,2560*480,1,stdout)!=1)return 6;
        fflush(stdout);XDestroyImage(image);
        next+=200000000ULL;struct timespec delay={next/1000000000ULL,next%1000000000ULL};clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&delay,NULL);
    }
    XCloseDisplay(d);return 0;
}
