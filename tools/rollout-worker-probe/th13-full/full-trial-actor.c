#define _POSIX_C_SOURCE 200809L
#include "x11_pixels.h"
#include "full_actor.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
extern int XQueryKeymap(Display *,char[32]);
static unsigned char keys[7];
static const int bits[7]={64,128,16,32,8,1,2};
static void control(Display*d,int old,int next) {
 for (int i=0;i<7;i++)if((old^next)&bits[i]) XTestFakeKeyEvent(d,keys[i],(next&bits[i])!=0,0);
 XSync(d,0);
}
static int held(Display*d){
 char map[32];XQueryKeymap(d,map);int v=0;
 for(int i=0;i<7;i++)if((unsigned char)map[keys[i]/8]&(1<<(keys[i]%8)))v|=bits[i];
 return v;
}
static int native(int fd,int32_t *out){
 for(int i=0;i<7;i++){if(pread(fd,out,84,0)==84 && out[0]>0 && out[0]==out[20])return 1;hold_ms(2);}
 return 0;
}
struct observation_label {
 int64_t start_ns, end_ns;
 int32_t before[15], after[15];
};
static void player_label(int fd,int32_t *row){
 if(fd<0 || pread(fd,row,15*sizeof(*row),0)!=15*(ssize_t)sizeof(*row) || row[0]!=row[14])memset(row,0,15*sizeof(*row));
}
static int mask(const int *action){
 int move=action[0],x=move%3-1,y=move/3-1;
 return (x<0?64:x>0?128:0)|(y<0?16:y>0?32:0)|(action[1]?8:0)|(action[2]?1:0)|(action[3]?2:0);
}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 uint64_t seed=strtoull(argv[2],NULL,10);
 int cap=atoi(argv[3]);if(cap<500||cap>4000||actor_init(argv[1],seed))return 3;
 Display*d=XOpenDisplay(NULL);if(!d)return 4;
 unsigned long syms[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78};
 for(int i=0;i<7;i++)keys[i]=XKeysymToKeycode(d,syms[i]);
 int fd=open("/capture/telemetry.bin",O_RDONLY);if(fd<0)return 5;
 FILE *devnull=fopen("/dev/null","wb"), *trace=fopen("steps.bin","wb"),*meta=fopen("events.tsv","w");
 if(!trace||!meta||!devnull)return 6;
 const int record_frames=access("/capture/lib/record-original-frames",F_OK)==0;
 FILE *pixels=record_frames?fopen("observations.u8","wb"):devnull;
 FILE *labels=record_frames?fopen("observation-labels.bin","wb"):NULL;
 int player=record_frames?open("/capture/xy-latest.bin",O_RDONLY):-1;
 if(!pixels||(record_frames&&!labels))return 6;
 int32_t pre[21],post[21]; unsigned char frame[112][96];int old=0,done=0,s2=0;
 if(!native(fd,pre)||pre[4]!=2||pre[5]!=2||pre[3]!=1||pre[6]!=0||pre[7]!=0||pre[2]!=2)return 7;
 int64_t beginning=now_ns();
 fprintf(meta,"step\tevent\tstage\tspares\tbombs\tstage_frame\tseconds\n");
 for(int n=0;!done&&n<25000;n++){
  if(!native(fd,pre))break;
  struct observation_label label={0};
  if(record_frames){
   if(player<0 && n%10==0)player=open("/capture/xy-latest.bin",O_RDONLY);
   label.start_ns=now_ns();player_label(player,label.before);
  }
  capture(d,pixels,devnull,frame);
  if(record_frames){
   player_label(player,label.after);label.end_ns=now_ns();
   if(fwrite(&label,sizeof(label),1,labels)!=1)return 9;
  }
  int a[4];actor_step(&frame[0][0],a);int next=mask(a);
  int64_t t=now_ns();control(d,old,next);old=next;int xheld=held(d);
  do {hold_ms(8);if(!native(fd,post))return 8;}
  while(post[2]==2 && post[3]==pre[3] && post[11]>=pre[11] && post[11]-pre[11]<4 && now_ns()-t<350000000);
  const float elapsed=(float)(now_ns()-t)/1e9f;
  /* Hold no keys while the original transitions or displays a terminal menu. */
  int consumed=post[10];
  int inactive=post[2]!=2;
  if(inactive){control(d,old,0);old=0;}
  double age=(double)(now_ns()-beginning)/1e9;
  int observed_loss=post[4]<pre[4],death=observed_loss?(pre[4]-post[4]):0;
  int extend=post[4]>pre[4]?(post[4]-pre[4]):0;
  int stage_up=post[3]==pre[3]+1 && post[3]>=2 && post[3]<=3;
  int spent=(post[4]==pre[4] && post[5]<pre[5])?pre[5]-post[5]:0;
  if(post[7]!=0){fprintf(meta,"%d\tunexpected_continue\t%d\t%d\t%d\t%d\t%.3f\n",n,post[3],post[4],post[5],post[11],age);break;}
  if(post[2]==1||post[4]<0){done=1;if(!death&&pre[4]==0)death=1;}
  if(inactive&&!done){
   for(int k=0;k<600&&post[2]==0;k++){hold_ms(16);native(fd,post);}
   if(post[2]==1||post[4]<0){done=1;if(!death&&pre[4]==0)death=1;}
   if(post[3]==pre[3]+1&&post[3]>=2&&post[3]<=3)stage_up=1;
  }
  /* Wait out the final-hit animation. Leave all seven buttons released so a menu cannot
     receive a confirmation. A recovered playable life remains part of the same credit. */
  if(done && post[4]<0 && post[2]==2){
   control(d,old,0);old=0;
   for(int k=0;k<3000&&post[4]<0&&post[2]==2;k++){hold_ms(18);if(!native(fd,post))return 8;}
   if(post[2]==2 && post[4]>=0){done=0;extend+=1;}
   else if(post[2]==2){fprintf(meta,"Unsettled final stock after timeout.\n");return 10;}
  }
  if(stage_up&&post[3]==2)s2=1;
  float reward=-death+extend+2*stage_up+.001f*elapsed-.03f*spent;
  float row[143]={0};memcpy(row,actor_features,110*sizeof(float));for(int i=0;i<4;i++){row[110+i]=a[i];row[114+i]=actor_logp[i];}
  row[118]=actor_value;row[119]=reward;row[120]=done;row[121]=(float)age;row[122]=next;row[123]=xheld;row[124]=consumed;
  row[125]=pre[3];row[126]=post[3];row[127]=pre[4];row[128]=post[4];row[129]=pre[5];row[130]=post[5];row[131]=post[6];row[132]=post[7];row[133]=post[2];row[134]=pre[11];row[135]=post[11];row[136]=death;row[137]=extend;row[138]=stage_up;row[139]=spent;row[140]=elapsed;row[141]=post[8];row[142]=post[1];
  if(fwrite(row,sizeof(row),1,trace)!=1)return 9;
  if(n%64==0||done||death||spent||stage_up){
   fprintf(meta,"%d\t%s\t%d\t%d\t%d\t%d\t%.3f\n",n,done?"terminal":stage_up?"stage":death?"loss":spent?"bomb":"pulse",post[3],post[4],post[5],post[11],age);
   fflush(meta);fflush(trace);
  }
  if(age>cap){fprintf(meta,"%d\ttechnical_timeout\t%d\t%d\t%d\t%d\t%.3f\n",n,post[3],post[4],post[5],post[11],age);break;}
 }
 control(d,old,0);fclose(trace);fclose(meta);
 if(record_frames){fclose(pixels);fclose(labels);if(player>=0)close(player);}
 fclose(devnull);close(fd);XCloseDisplay(d);printf("terminal=%d stage2=%d\n",done,s2);return done?0:10;
}
