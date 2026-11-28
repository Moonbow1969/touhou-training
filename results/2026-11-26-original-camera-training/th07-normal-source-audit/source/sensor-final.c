#define main probe_main
#include "probe-clock.c"
#undef main
#include <dlfcn.h>
#include <sys/time.h>
typedef struct {int w,h,xoffset,format;char*data;int order,unit,bit_order,pad,depth,stride,bpp;unsigned long rm,gm,bm;} XI;
static unsigned adr[15000];
static uint32_t vals[15000];
static unsigned char pixels[112*96];
int main(int argc,char**argv){
    const unsigned clockaddr=argc>1?strtoul(argv[1],0,16):0x11c9b24;
    const int seconds=argc>2?atoi(argv[2]):210;
    int fd=gamefd(),n=0; if(fd<0)return 1;
    FILE *a=fopen("/capture/all.tsv","r");while(n<15000&&fscanf(a,"%x",adr+n)==1)n++;fclose(a);
    void*lib=dlopen("libX11.so.6",RTLD_NOW);if(!lib){fprintf(stderr,"%s",dlerror());return 2;}
    void*(*op)(const char*)=dlsym(lib,"XOpenDisplay");unsigned long(*root)(void*)=dlsym(lib,"XDefaultRootWindow");
    XI*(*snap)(void*,unsigned long,int,int,unsigned,unsigned,unsigned long,int)=dlsym(lib,"XGetImage");
    int(*freei)(XI*)=dlsym(lib,"XDestroyImage");
    void*d=op(0);if(!d||!root||!snap||!freei)return 3;
    unsigned long w=root(d);
    FILE *f=fopen("/capture/final-words.bin","wb"),*p=fopen("/capture/final-pix.bin","wb"),*m=fopen("/capture/final-clock.csv","w");
    time_t end=time(NULL)+seconds;unsigned seen=0xffffffff;int cnt=0;
    while(time(NULL)<end){
        unsigned c=0,b=0;if(pread(fd,&c,4,clockaddr)!=4)return 4;
        if(seen!=0xffffffff && seen>8500 && c+400<seen)break;
        if(c==seen||c%4||c>30000){usleep(750);continue;}
        XI*i=snap(d,w,32,16,384,448,~0UL,2);
        if(!i||i->bpp!=32||i->rm!=0xff0000||i->bm!=255)return 6;
        for(int j=0;j<n;j++)if(pread(fd,&vals[j],4,adr[j])!=4)return 7;
        if(pread(fd,&b,4,clockaddr)!=4)return 4;
        if(b!=c){freei(i);continue;}
        
        for(int y=0;y<112;y++)for(int x=0;x<96;x++){
            int sum=0;
            for(int v=0;v<4;v++){
                int row=0;unsigned char *src=(unsigned char*)i->data+(y*4+v)*i->stride+x*16;
                for(int u=0;u<4;u++,src+=4)row+=(19595*src[2]+38470*src[1]+7471*src[0]+32768)>>16;
                sum+=(row+2)/4;
            }
            pixels[y*96+x]=(sum+2)/4;
        }
        struct timeval tv;gettimeofday(&tv,0);
        fwrite(vals,4,n,f);fwrite(pixels,1,sizeof(pixels),p);fprintf(m,"%u,%ld.%06ld\n",c,(long)tv.tv_sec,(long)tv.tv_usec);
        if(++cnt%16==0){fflush(f);fflush(p);fflush(m);}
        seen=c;freei(i);
    }
    fclose(f);fclose(p);fclose(m);return 0;
}
