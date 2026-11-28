/* Offline watchdog test. Both PIDs are children created by this fixture. No game is opened. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
extern long long mono_ns(void);
extern int freeze_game(void);
extern void protect_parent(int);
#include <sys/prctl.h>

static int stopped(pid_t pid) {
    char path[128],line[128];snprintf(path,sizeof(path),"/proc/%d/status",pid);
    FILE *f=fopen(path,"r");if(!f)return -1;
    int s=0;while(fgets(line,sizeof(line),f))if(!strncmp(line,"State:",6)){s=strchr(line,'T')!=NULL;break;}
    fclose(f);return s;
}
static int probe(int interrupt) {
    int ready[2],permission[2];if(pipe(ready)||pipe(permission))return 1;
    pid_t dummy=fork();
    if(dummy==0){
        prctl(PR_SET_PTRACER,PR_SET_PTRACER_ANY,0,0,0);
        if(write(permission[1],"y",1)!=1)_exit(6);
        for(;;)pause();
    }
    char ch;if(read(permission[0],&ch,1)!=1)return 2;
    pid_t controller=fork();
    if(controller==0){
        char name[128];snprintf(name,sizeof(name),"/proc/%d/mem",dummy);
        int fd=open(name,O_RDONLY);if(fd<0)_exit(3);
        protect_parent(fd);
        if(!freeze_game())_exit(4);
        if(write(ready[1],"t",1)!=1)_exit(7);
        for(;;)pause();
    }
    close(ready[1]);if(read(ready[0],&ch,1)!=1){kill(dummy,SIGKILL);return 5;}
    int initially=stopped(dummy),status;
    long long start=mono_ns();
    if(interrupt)kill(controller,SIGKILL);
    waitpid(controller,&status,0);
    while(stopped(dummy)==1 && mono_ns()-start<2500000000ll)usleep(1000);
    int resumed=stopped(dummy)==0;
    printf("%s initially_stopped=%d resumed=%d elapsed_ms=%.2f\n",
           interrupt?"controller_kill":"controller_stall",initially,resumed,(mono_ns()-start)/1e6);
    kill(dummy,SIGCONT);kill(dummy,SIGTERM);waitpid(dummy,&status,0);
    return !(initially==1&&resumed);
}
int main(void){if(probe(0))return 1;return probe(1);}
