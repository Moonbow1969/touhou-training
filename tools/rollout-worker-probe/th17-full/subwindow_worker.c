/* One physical trial camera per four stage ticks. A second command uses the same image. */
#define main existing_wall_worker_main
#include "worker.c"
#undef main
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int playable(State s,int rank){return s.stage>0&&s.stage<=3&&s.rank==rank&&s.lives>=0&&!s.continues;}
static State reach(State start,int ticks,int rank){
    uint64_t began=now_ns();
    struct timespec wait={0,300000};
    for(;;){
        State q=read_state();
        if(!playable(q,rank)||q.stage!=start.stage||q.tick>=start.tick+ticks)return q;
        if(now_ns()-began>UINT64_C(12000000000))fail("Original stage clock stalled");
        nanosleep(&wait,NULL);
    }
}
static void put_frame(int fd,const uint8_t *p,size_t n){
    while(n){ssize_t res=send(fd,p,n,MSG_NOSIGNAL);if(res<=0)fail("Local visual actor connection failed");n-=(size_t)res;p+=res;}
}
static unsigned checked_mask(const uint8_t *command){int a[4];for(int j=0;j<4;j++){a[j]=command[j];if(a[j]>(j?1:8))fail("Invalid action");}return mask(a);}
int main(int argc,char **argv){
    if(argc!=4||strlen(argv[1])>=sizeof(((struct sockaddr_un*)0)->sun_path))return 2;
    int rank=atoi(argv[2]),limit=atoi(argv[3]);if((rank!=2&&rank!=3)||(limit!=0&&limit!=96))return 2;
    initialize();Display *d=XOpenDisplay(NULL);if(!d)fail("No private X");
    const char *keynames[]={"Left","Right","Up","Down","Shift_L","z","x"};
    for(int i=0;i<7;i++)codes[i]=XKeysymToKeycode(d,XStringToKeysym(keynames[i]));
    int f=open("live.state",O_RDONLY);if(f<0)fail("Original clock missing");
    shared=mmap(NULL,sizeof(*shared),PROT_READ,MAP_SHARED,f,0);if(shared==MAP_FAILED)fail("Original clock not readable");
    int sock=socket(AF_UNIX,SOCK_STREAM,0);struct sockaddr_un path={.sun_family=AF_UNIX};strcpy(path.sun_path,argv[1]);
    if(connect(sock,(struct sockaddr*)&path,sizeof path)<0)fail("Visual model unavailable");
    FILE *steps=fopen("subwindow-wire.tsv","wx"),*obs=fopen("observations.u8","wbx");
    if(!steps||!obs)fail("Cannot preserve original wire");
    fputs("group\tstage\timage_tick\timage_end_tick\tspares_before\tbombs_before\tfirst_requested\tfirst_observed\tfirst_prior_tick\tfirst_applied_tick\tsecond_requested\tsecond_observed\tsecond_prior_tick\tsecond_applied_tick\tend_stage\tend_tick\tend_spares\tend_bombs\timage_ns\tfirst_onset_ns\tsecond_onset_ns\tend_ns\n",steps);
    unsigned held=0;State origin,post={0};struct timespec wait={0,500000};uint64_t start=now_ns();
    do{origin=read_state();if(playable(origin,rank)&&origin.stage==1&&origin.tick>=2&&origin.tick<=150&&origin.lives==2&&origin.bombs==3)break;
       if(now_ns()-start>UINT64_C(180000000000))fail("No fresh original game");
       nanosleep(&wait,NULL);
    }while(1);
    int index=0,terminal=0;
    for(;limit==0||index<limit;index++){
        State pre=read_state();
        if(!playable(pre,rank)){terminal=pre.lives==-1;break;}
        if(now_ns()-start>UINT64_C(1800000000000))fail("No actual terminal");
        uint8_t frame[112*96],cmd[2][4];uint64_t image_ns=now_ns();capture(d,frame);State image_end=read_state();
        put_frame(sock,frame,sizeof frame);
        if(recv(sock,cmd,sizeof(cmd),MSG_WAITALL)!=(ssize_t)sizeof(cmd))fail("Incomplete learned actions");
        unsigned k0=checked_mask(cmd[0]),k1=checked_mask(cmd[1]);
        State a0=read_state(),b0=a0,a1=a0,b1=a0;int observed[2]={-1,-1};uint64_t first=0,second=0;post=a0;
        if(playable(a0,rank)&&a0.stage==pre.stage){
            first=now_ns();observed[0]=(int)apply(d,held,k0);held=k0;b0=read_state();
            if(observed[0]!=(int)k0)fail("Original X first key mismatch");
            a1=reach(pre,2,rank);b1=a1;post=a1;
            if(playable(a1,rank)&&a1.stage==pre.stage){
                second=now_ns();observed[1]=(int)apply(d,held,k1);held=k1;b1=read_state();
                if(observed[1]!=(int)k1)fail("Original X second key mismatch");
                post=reach(pre,4,rank);
            }
        }
        uint64_t end=now_ns();
        fwrite(frame,sizeof(frame),1,obs);
        fprintf(steps,"%d\t%d\t%d\t%d\t%d\t%d\t%u\t%d\t%d\t%d\t%u\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%llu\t%llu\t%llu\t%llu\n",index,pre.stage,pre.tick,image_end.tick,pre.lives,pre.bombs,k0,observed[0],a0.tick,b0.tick,k1,observed[1],a1.tick,b1.tick,post.stage,post.tick,post.lives,post.bombs,(unsigned long long)image_ns,(unsigned long long)first,(unsigned long long)second,(unsigned long long)end);
        if(!playable(post,rank)){terminal=post.lives==-1;index++;break;}
    }
    if(apply(d,held,0)!=0)fail("Physical keys not released");
    fclose(steps);fclose(obs);close(sock);XCloseDisplay(d);
    printf("groups=%d terminal=%d limit=%d\n",index,terminal,limit);return (limit==96&&index==96)||(limit==0&&terminal)?0:8;
}
