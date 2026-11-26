#define _POSIX_C_SOURCE 200809L
#include "x11_pixels.h"
#include <fcntl.h>
static int get(int f,int32_t*r){return pread(f,r,60,0)==60&&r[0]>0&&r[0]==r[14];}
int main(int argc,char**argv){
 if(argc!=2)return 1;int cap=atoi(argv[1]),fd=open("/capture/xy-latest.bin",O_RDONLY);if(fd<0)return 2;
 Display*d=XOpenDisplay(NULL);FILE *f=fopen("frames.u8","wb"),*labels=fopen("native-labels.bin","wb"),*tmp=fopen("/dev/null","wb");
 if(!d||!f||!labels||!tmp)return 3;
 int32_t before[15],after[15],last=-1;int64_t wall=now_ns();
 for(int i=0;i<cap&&now_ns()-wall<300000000000LL;){
  if(!get(fd,before)){hold_ms(6);continue;}
  if(before[4]<0||before[13]==1)break;
  if(before[13]!=2||before[2]==last){hold_ms(12);continue;}
  unsigned char frame[112][96];int64_t t0=now_ns();capture(d,tmp,tmp,frame);int64_t t1=now_ns();
  if(!get(fd,after))continue;
  fwrite(frame,sizeof frame,1,f);fwrite(&t0,8,1,labels);fwrite(&t1,8,1,labels);fwrite(before,60,1,labels);fwrite(after,60,1,labels);last=before[2];i++;
  if(i%32==0){fflush(f);fflush(labels);}hold_ms(95);
 }
 fclose(f);fclose(labels);fclose(tmp);close(fd);XCloseDisplay(d);return 0;
}
