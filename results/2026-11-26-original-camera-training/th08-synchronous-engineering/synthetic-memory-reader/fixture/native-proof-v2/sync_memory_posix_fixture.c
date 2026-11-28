#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <string.h>
static char status(int p){char name[100],line[100];snprintf(name,sizeof(name),"/proc/%d/status",p); FILE*f=fopen(name,"r");if(!f)return '?';char state='?';while(fgets(line,sizeof(line),f))if(sscanf(line,"State: %c",&state)==1)break;fclose(f);return state;}
int main(int argc,char**argv){if(argc!=3)return 1; int pid=atoi(argv[1]);int fd=open("/capture/tick.bin",O_RDONLY);if(fd<0)return 2;
 volatile unsigned *tick=mmap(NULL,4096,PROT_READ,MAP_SHARED,fd,0);if(tick==MAP_FAILED)return 3;
 printf("phase,trial,state_proven_tick,last_tick,delta_native,observed_T,observed_t,observed_S,observed_R,observed_other\n");
 for(int j=0;j<100;j++){if(kill(pid,SIGSTOP))return 4;char ch='?';for(int k=0;k<400;k++){ch=status(pid);if(ch=='T')break;usleep(50);}if(ch!='T'){kill(pid,SIGCONT);return 5;}
  unsigned a=*tick;int states[256]={0};for(int i=0;i<45;i++){states[(unsigned char)status(pid)]++;usleep(200);} unsigned z=*tick;kill(pid,SIGCONT);
  printf("%s,%d,%u,%u,%u,%d,%d,%d,%d,%d\n",argv[2],j,a,z,z-a,states['T'],states['t'],states['S'],states['R'],45-states['T']-states['t']-states['S']-states['R']);fflush(stdout);usleep(2500);
 } return 0;}
