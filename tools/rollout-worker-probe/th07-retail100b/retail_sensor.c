#define main legacy_main
#include "native_input.c"
#undef main
#include <dlfcn.h>
#include <math.h>
#include <stdlib.h>
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
  int expected=(getenv("TH07_STAGE") ? *getenv("TH07_STAGE") - 48 : 0);
  if(expected!=1&&expected!=2)return 21;
  time_t end=time(NULL)+570;unsigned seen=0,rows=0,high=0;int departed=0;
  #define R(addr,dst) (pread(fd,&(dst),sizeof(dst),(addr))==(int)sizeof(dst))
  const unsigned position[]={0x4be408,0x4be420,0x4be42c,0x4be438,0x4be444};
  for(unsigned j=0;j<5;j++)fprintf(cand,"%x\t1\n",position[j]);
  fclose(cand);
  while(time(NULL)<end){
    uint32_t manager=0,stage=0,t=0,t1=0,stage1=0,mgr1=0,data=0,stageblk=0,cursor=0,cursor1=0;
    uint32_t stage_timer=0,flags=0;uint16_t now=0,old=0,now1=0,old1=0;
    if(!R(0x4b9e48,manager)||!R(0x62f85c,stage))return 22;
    if(seen>1000 && (stage!=(unsigned)expected||!manager)){departed=1;break;}
    if(stage!=(unsigned)expected||!manager||!R(manager,t)||!t||t%4||t==seen||t>38000){usleep(500);continue;}
    if(seen>1000 && t+400<seen){departed=1;break;}
    if(!R(manager+4,data)||!data||!R(data+28+4*(expected-1),stageblk)||!stageblk||!R(manager+0x84,cursor)||!cursor){usleep(500);continue;}
    if(!R(0x4b9e50,now)||!R(0x4b9e58,old)||!R(0x62f858,stage_timer)||!R(0x62f648,flags))return 23;
    for(unsigned j=0;j<5;j++)if(pread(fd,values[j],8,position[j])!=8)return 24;
    XI*i=snap(d,w,0,0,640,480,~0UL,2);
    if(!i||i->bpp!=32||i->rm!=0xff0000||i->bm!=255||i->stride!=2560)return 25;
    if(!R(0x4b9e48,mgr1)||mgr1!=manager||!R(manager,t1)||!R(manager+0x84,cursor1)||!R(0x62f85c,stage1)||!R(0x4b9e50,now1)||!R(0x4b9e58,old1)){freei(i);continue;}
    if(t!=t1||now!=now1||old!=old1||cursor!=cursor1||stage!=stage1){freei(i);continue;}
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
      int sum=0;
      for(int v=0;v<4;v++){int row=0;unsigned char *src=(unsigned char*)i->data+(16+y*4+v)*i->stride+(32+x*4)*4;
        for(int u=0;u<4;u++,src+=4)row+=(19595*src[2]+38470*src[1]+7471*src[0]+32768)>>16;sum+=(row+2)/4;}
      gray[y*96+x]=(sum+2)/4;
    }
    int full=(t%1200==0||t%1200==1196||t==4||t==8);
    if(full){if(gzwrite_(rgb,&t,4)!=4||gzwrite_(rgb,i->data,640*480*4)!=640*480*4)return 13;high++;}
    if(gzwrite_(video,gray,sizeof(gray))!=sizeof(gray)||fwrite(values,8,5,f)!=5)return 14;
    fprintf(csv,"%u,%u,%u,%d,%u,%u,%u,%u,%u,%u\n",t,now,old,full,stage,stage_timer,flags,cursor,stageblk,manager);
    if(++rows%64==0){fflush(csv);fflush(f);}
    seen=t;freei(i);
  }
  fclose(f);fclose(csv);if(gzclose_(rgb)||gzclose_(video))return 15;
  fprintf(stderr,"last=%u rows=%u color=%u\n",seen,rows,high);return departed?0:16;
}



