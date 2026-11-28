/* Used only by opt-in engineering builds. Xvfb, wineserver and the native reader stay runnable. */
#include <dirent.h>
#include <ctype.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
static int sync_gamefd=-1,sync_guard=-1,game_stopped=0;
static pid_t sync_pid;
static FILE *freeze_receipt;
static int sync_decimal(const char *s){int v=0;while(*s>='0'&&*s<='9'){if(v>100000000)return -1;v=v*10+*s++-'0';}return v;}
static int sync_signal(int sig){return (int)syscall(SYS_pidfd_send_signal,sync_gamefd,sig,NULL,0);}
static int stopped(void){
    char path[80],line[128];snprintf(path,sizeof(path),"/proc/%d/status",sync_pid);
    FILE *f=fopen(path,"r");if(!f)return 0;
    int hit=0;while(fgets(line,sizeof(line),f))if(!strncmp(line,"State:",6)){hit=strchr(line,'T')!=NULL;break;}
    fclose(f);return hit;
}
static void sync_resume(void){
    if(game_stopped){sync_signal(SIGCONT);game_stopped=0;if(sync_guard>=0){char off='R';(void)write(sync_guard,&off,1);}}
}
static void emergency_resume(int sig){if(sync_gamefd>=0)sync_signal(SIGCONT);_exit(128+sig);}
static int bind_sync_game(void){
    DIR *d=opendir("/proc");if(!d)return 0;struct dirent *e;int count=0;
    while((e=readdir(d))){
        if(!isdigit((unsigned char)e->d_name[0]))continue;
        char path[320],comm[64];snprintf(path,sizeof(path),"/proc/%s/comm",e->d_name);
        FILE *f=fopen(path,"r");if(!f)continue;
        if(fgets(comm,sizeof(comm),f)&&!strcmp(comm,"th08tr.exe\n")){sync_pid=(pid_t)sync_decimal(e->d_name);count++;}
        fclose(f);
    }
    closedir(d);if(count!=1)return 0;
    sync_gamefd=(int)syscall(SYS_pidfd_open,sync_pid,0);if(sync_gamefd<0)return 0;
    int channel[2];if(pipe(channel))return 0;
    pid_t owner=getpid(),child=fork();if(child<0)return 0;
    if(child==0){
        close(channel[1]);close(STDIN_FILENO);close(STDOUT_FILENO);
        struct pollfd p={.fd=channel[0],.events=POLLIN};int armed=0;
        for(;;){
            int ready=poll(&p,1,armed?800:-1);if(ready<0&&errno==EINTR)continue;
            if(ready<=0){sync_signal(SIGCONT);if(getppid()==owner)kill(owner,SIGTERM);_exit(25);}
            char msg;int got=read(channel[0],&msg,1);if(got!=1){sync_signal(SIGCONT);_exit(0);}
            armed=(msg=='S');
        }
    }
    close(channel[0]);sync_guard=channel[1];atexit(sync_resume);
    signal(SIGTERM,emergency_resume);signal(SIGPIPE,emergency_resume);signal(SIGINT,emergency_resume);
    freeze_receipt=fopen("/capture/frozen.tsv","wx");if(!freeze_receipt)return 0;
    fprintf(freeze_receipt,"index pid stage tick stop_start_ns stopped_ns heartbeat_ns camera_ns cmd_ns ack_ns resume_ns serial_start serial_confirmed\n");
    return 1;
}
static unsigned frozen_first_serial,frozen_proven_serial;
static int64_t freeze_start_ns,freeze_stop_ns,freeze_proof_ns;
static int freeze_at(unsigned stage,int tick){
    frozen_first_serial=tick_source->serial;
    if(write(sync_guard,"S",1)!=1)return 0;
    freeze_start_ns=now_ns();
    if(sync_signal(SIGSTOP)!=0){(void)write(sync_guard,"R",1);return 0;}
    game_stopped=1;
    while(!stopped()&&now_ns()-freeze_start_ns<30000000)usleep(100);
    if(!stopped()){sync_resume();return 0;}
    freeze_stop_ns=now_ns();unsigned serial=tick_source->serial;
    /* Require a fresh valid process-memory sample from clock.exe after the game stopped. */
    for(int i=0;i<200 && now_ns()-freeze_stop_ns<30000000;i++){
        unsigned a=tick_source->serial;unsigned actual_stage=tick_source->stage,actual_tick=tick_source->tick;
        unsigned b=tick_source->serial;
        if(a==b&&!(a&1u)&&a>serial){
            if(actual_stage!=stage||(int)actual_tick!=tick){sync_resume();return 0;}
            frozen_proven_serial=a;freeze_proof_ns=now_ns();return 1;
        }
        usleep(150);
    }
    sync_resume();return 0;
}
static int frozen_matches(unsigned stage,int tick){
    return stopped()&&tick_source->stage==stage&&(int)tick_source->tick==tick;
}
