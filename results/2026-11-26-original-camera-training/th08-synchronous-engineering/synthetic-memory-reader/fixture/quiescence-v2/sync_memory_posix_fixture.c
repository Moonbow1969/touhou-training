#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <string.h>
#include "quiet_reads.h"
static char status(int p){char name[100],line[100];snprintf(name,sizeof(name),"/proc/%d/status",p); FILE*f=fopen(name,"r");if(!f)return '?';char state='?';while(fgets(line,sizeof(line),f))if(sscanf(line,"State: %c",&state)==1)break;fclose(f);return state;}
int main(int argc,char**argv){if(argc!=3)return 1; int pid=atoi(argv[1]);int fd=open("/capture/tick.bin",O_RDONLY);if(fd<0)return 2;
 volatile unsigned *tick=mmap(NULL,4096,PROT_READ,MAP_SHARED,fd,0);if(tick==MAP_FAILED)return 3;
 int controlled=!strcmp(argv[2],"quiescent");_Atomic unsigned *reads=NULL;if(controlled){if(!quiet_open(0))return 6;int f=open("/capture/reads.bin",O_RDONLY);reads=mmap(NULL,4096,PROT_READ,MAP_SHARED,f,0);if(f<0||reads==MAP_FAILED)return 7;}
 printf("phase,trial,state_proven_tick,last_tick,delta_native,observed_T,observed_t,observed_S,observed_R,observed_other\n");
 for(int j=0;j<100;j++){if(kill(pid,SIGSTOP))return 4;char ch='?';for(int k=0;k<400;k++){ch=status(pid);if(ch=='T')break;usleep(50);}if(ch!='T'){kill(pid,SIGCONT);return 5;}
  if(controlled){unsigned n=reads[0];int64_t end=quiet_now()+50000000;while(reads[0]==n&&quiet_now()<end)quiet_nap();if(reads[0]==n||!quiet_request()){kill(pid,SIGCONT);return 8;}if(reads[1]!=*tick){kill(pid,SIGCONT);return 9;}}
  unsigned a=*tick;int states[256]={0};for(int i=0;i<45;i++){states[(unsigned char)status(pid)]++;usleep(200);} unsigned z=*tick;if(controlled){unsigned n=reads[0];quiet_release();int64_t end=quiet_now()+50000000;while(reads[0]==n&&quiet_now()<end)quiet_nap();if(reads[0]==n||!quiet_request()||reads[1]!=a||*tick!=a){kill(pid,SIGCONT);return 10;}quiet_release();}kill(pid,SIGCONT);
  printf("%s,%d,%u,%u,%u,%d,%d,%d,%d,%d\n",argv[2],j,a,z,z-a,states['T'],states['t'],states['S'],states['R'],45-states['T']-states['t']-states['S']-states['R']);fflush(stdout);usleep(2500);
 } return 0;}

