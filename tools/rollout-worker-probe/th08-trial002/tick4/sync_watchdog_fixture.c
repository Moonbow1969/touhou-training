#define _GNU_SOURCE
#include "x11_pixels.h"
#include "gate.h"
#include "sync_gate.h"
#include <sys/prctl.h>
#include <sys/wait.h>
/* Run inside a fresh empty PID namespace. The only TH08-named process is our own child. */
int main(int argc,char **argv){
    if(argc!=2)return 2;
    int arm[2],ready[2];if(pipe(arm)||pipe(ready))return 2;
    pid_t dummy=fork();
    if(dummy==0){prctl(PR_SET_NAME,"th08tr.exe",0,0,0);write(ready[1],"D",1);for(;;)pause();}
    char msg;if(read(ready[0],&msg,1)!=1)return 3;
    pid_t controller=fork();
    if(controller==0){
        if(!bind_sync_game()||sync_pid!=dummy)_exit(4);
        if(write(sync_guard,"S",1)!=1||sync_signal(SIGSTOP))_exit(5);
        game_stopped=1;while(!stopped())usleep(100);
        write(arm[1],"F",1);for(;;)pause();
    }
    close(arm[1]);if(read(arm[0],&msg,1)!=1)return 6;
    sync_pid=dummy;
    if(!stopped())return 7;
    int64_t start=now_ns();
    if(!strcmp(argv[1],"orphan"))kill(controller,SIGKILL);
    int status=0;waitpid(controller,&status,0);
    while(stopped()&&now_ns()-start<2000000000)usleep(1000);
    int released=!stopped();printf("%s resumed=%d milliseconds=%.3f\n",argv[1],released,(now_ns()-start)/1e6);
    kill(dummy,SIGCONT);kill(dummy,SIGTERM);waitpid(dummy,NULL,0);
    return released?0:8;
}
