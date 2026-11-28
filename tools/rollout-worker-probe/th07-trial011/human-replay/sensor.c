#define main legacy_main
#include "native_input.c"
#undef main
#include <dlfcn.h>
#include <math.h>
#include <sys/time.h>
typedef struct {int w,h,xoffset,format;char*data;int order,unit,bit_order,pad,depth,stride,bpp;unsigned long rm,gm,bm;} XI;
#define LIMIT 4096
struct Candidate { unsigned addr; float scale; } candidates[LIMIT];
static uint32_t values[LIMIT][2];
static unsigned char gray[112*96];
int main(void){
  int fd=open_game();if(fd<0)return 1;
  void *lib=dlopen("libX11.so.6",RTLD_NOW),*z=dlopen("libz.so.1",RTLD_NOW);
  if(!lib||!z)return 2;
  void*(*op)(const char*)=dlsym(lib,"XOpenDisplay");unsigned long(*root)(void*)=dlsym(lib,"XDefaultRootWindow");
  XI*(*snap)(void*,unsigned long,int,int,unsigned,unsigned,unsigned long,int)=dlsym(lib,"XGetImage");
  int(*freei)(XI*)=dlsym(lib,"XDestroyImage");void*(*gzopen_)(const char*,const char*)=dlsym(z,"gzopen");
  int(*gzwrite_)(void*,const void*,unsigned)=dlsym(z,"gzwrite");int(*gzclose_)(void*)=dlsym(z,"gzclose");
  void*d=op(0);if(!d||!snap||!freei||!gzopen_||!gzwrite_||!gzclose_)return 3;
  unsigned long w=root(d);FILE*f=fopen("/capture/native.bin","wb"),*csv=fopen("/capture/clock.csv","w"),*cand=fopen("/capture/candidates.tsv","w");
  void* video=gzopen_("/capture/gray.raw.gz","wb1");void* rgb=gzopen_("/capture/full-color.bgra.gz","wb1");
  if(!f||!csv||!cand||!video||!rgb)return 4;
  time_t end=time(NULL)+390;unsigned seen=0,n=0,rows=0,high=0;
  while(time(NULL)<end){
    uint32_t t,t1,k0,k1,kp,p1;
    if(pread(fd,&t,4,0x11c9ae8)!=4)return 5;
    if(seen>8500&&t+400<seen)break;
    if(!t||t%4||t==seen||t>26000){usleep(500);continue;}
    if(!n){
       if(t>120){fprintf(stderr,"No opening native clock: %u",t);return 6;}
       const unsigned qualified[]={0x1280e58,0x1280e70,0x1280e7c,0x1280e88,0x1280e94};
       for(unsigned i=0;i<sizeof(qualified)/sizeof(*qualified);i++){
           candidates[n]=(struct Candidate){qualified[i],1};
           fprintf(cand,"%x\t1\n",qualified[i]);n++;
       }
       fclose(cand);fprintf(stderr,"open t=%u candidates=%u\n",t,n);fflush(stderr);if(!n)return 9;
    }
    if(pread(fd,&k0,4,0x11c9d24)!=4||pread(fd,&kp,4,0x11c9d2c)!=4)return 10;
    for(unsigned j=0;j<n;j++)if(pread(fd,values[j],8,candidates[j].addr)!=8)return 10;
    XI*i=snap(d,w,0,0,640,480,~0UL,2);
    if(!i||i->bpp!=32||i->rm!=0xff0000||i->bm!=255||i->stride!=2560)return 11;
    if(pread(fd,&t1,4,0x11c9ae8)!=4||pread(fd,&k1,4,0x11c9d24)!=4||pread(fd,&p1,4,0x11c9d2c)!=4)return 12;
    if(t!=t1||k0!=k1||kp!=p1){freei(i);continue;}
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
      int sum=0;
      for(int v=0;v<4;v++){int row=0;unsigned char *src=(unsigned char*)i->data+(16+y*4+v)*i->stride+(32+x*4)*4;
        for(int u=0;u<4;u++,src+=4)row+=(19595*src[2]+38470*src[1]+7471*src[0]+32768)>>16;sum+=(row+2)/4;}
      gray[y*96+x]=(sum+2)/4;
    }
    int full=(t%240==0||t%240==236);
    if(full){if(gzwrite_(rgb,&t,4)!=4||gzwrite_(rgb,i->data,640*480*4)!=640*480*4)return 13;high++;}
    if(gzwrite_(video,gray,sizeof(gray))!=sizeof(gray)||fwrite(values,8,n,f)!=n)return 14;
    fprintf(csv,"%u,%u,%u,%d\n",t,k0&65535,kp&65535,full);
    if(++rows%64==0){fflush(csv);fflush(f);}
    seen=t;freei(i);
  }
  fclose(f);fclose(csv);if(gzclose_(rgb)||gzclose_(video))return 15;
  fprintf(stderr,"last=%u rows=%u color=%u\n",seen,rows,high);return seen>8500?0:16;
}


