#define main old_native_main
#include "native_input.c"
#undef main
#include <stdlib.h>
int main(void){
 while(access("/capture/menu-released",F_OK)!=0)usleep(2000);
 int fd=open_game();if(fd<0)return 1;
 FILE*f=fopen("/capture/spell-phase.csv","wx");if(!f)return 2;
 fputs("ns,stage_tick,supervisor,capturing,active,capture_score,spell_index,spell_timer,native_keys,finished_stage,spell_count\n",f);
 int prev_t=-1,prev_state=-1, nonplay=0;
 long long end=mono_ns()+160000000000LL;
 while(mono_ns()<end){
  int t=0,t2=0,super=0,block[10]={0},keys=0;
  int gui=0,result=0,stageclear=-1,count=-1;
  if(pread(fd,&gui,4,0x49fbf8)==4 && gui)pread(fd,&stageclear,4,(off_t)(unsigned)gui+0x209b4);
  if(pread(fd,&result,4,0x626278)==4 && result)pread(fd,&count,4,(off_t)(unsigned)result+0x1c);
  if(pread(fd,&t,4,0x62f858)!=4||pread(fd,block,sizeof(block),0x9a9b00+0x9545c4)!=(ssize_t)sizeof(block)||pread(fd,&super,4,0x575aa8)!=4||pread(fd,&keys,2,0x4b9e50)!=2||pread(fd,&t2,4,0x62f858)!=4)break;
  if(t==t2 && (t!=prev_t||super!=prev_state)){
   fprintf(f,"%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",mono_ns(),t,super,block[0],block[1],block[2],block[5],block[9],keys,stageclear,count);
   prev_t=t;prev_state=super;
   if((t%15)==0||super!=2)fflush(f);
  }
  if(super==2)nonplay=0;else if(++nonplay>500&&prev_t>30)break;
  usleep(7000);
 }
 fclose(f);close(fd);return 0;
}
