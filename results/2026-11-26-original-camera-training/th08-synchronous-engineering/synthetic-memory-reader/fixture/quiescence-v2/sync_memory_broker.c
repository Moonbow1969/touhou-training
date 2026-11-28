#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include "quiet_reads.h"
/* Fixture feeds the Wine reader exactly one q per loop, mirroring original bracket.c. */
int main(void){if(!quiet_open(1))return 2;int in[2],out[2];if(pipe(in)||pipe(out))return 3;pid_t kid=fork();if(kid<0)return 4;
 if(!kid){dup2(in[0],0);dup2(out[1],1);close(in[0]);close(in[1]);close(out[0]);close(out[1]);execlp("wine","wine","Z:/capture/proofread.exe","broker",(char*)0);_exit(5);}
 close(in[0]);close(out[1]);FILE*reader=fdopen(out[0],"r");char line[80];int fd=open("/capture/reads.bin",O_RDWR|O_CREAT|O_EXCL,0644);ftruncate(fd,4096);_Atomic unsigned *m=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
 while(quiet_between_reads()&&write(in[1],"q",1)==1&&fgets(line,sizeof(line),reader)){unsigned word;if(sscanf(line,"%u",&word)!=1)return 6;atomic_store(&m[1],word);atomic_fetch_add(&m[0],1);struct timespec nap={0,850000};nanosleep(&nap,NULL);}return 1;}
