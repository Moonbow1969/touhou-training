#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

static int64_t stamp(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*INT64_C(1000000000)+t.tv_nsec;
}
#include <fcntl.h>
#include <stdatomic.h>
#include <sys/mman.h>
typedef struct { _Atomic unsigned serial,stage,tick; _Atomic int64_t upper_ns; } TickShared;
int main(void) {
    int fd=open("/capture/current.tick",O_RDWR|O_CREAT|O_EXCL,0644);
    if(fd<0||ftruncate(fd,sizeof(TickShared)))return 6;
    TickShared *shared=mmap(0,sizeof(*shared),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(shared==MAP_FAILED)return 6;

    int requests[2], responses[2];
    if(pipe(requests) || pipe(responses)) return 2;
    pid_t child=fork();
    if(child<0)return 2;
    if(child==0) {
        dup2(requests[0],0);dup2(responses[1],1);
        close(requests[0]);close(requests[1]);close(responses[0]);close(responses[1]);
        execlp("wine","wine","Z:/capture/clock.exe",(char *)0);_exit(3);
    }
    close(requests[0]);close(responses[1]);
    FILE *from=fdopen(responses[0],"r");
    FILE *out=fopen("/capture/clock.tsv","wx");
    if(!from || !out)return 4;
    fprintf(out,"before_ns after_ns pid base object stage_index tick mode flags raw word previous info origin cursor saved_prior saved_current player focus x y buffer_read stable\n");
    const int64_t begin=stamp(); char line[512];
    while(stamp()-begin<INT64_C(1800000000000)) {
        int64_t before=stamp();
        if(write(requests[1],"q",1)!=1 || !fgets(line,sizeof(line),from))break;
        int64_t after=stamp();
        unsigned pid,base,obj,stage,tick,mode,flags,raw,word,prev,info,origin,cursor,prior,current,player,focus,valid_buffer,stable;
        float x,y;
        if(sscanf(line,"%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %f %f %u %u",
                &pid,&base,&obj,&stage,&tick,&mode,&flags,&raw,&word,&prev,&info,&origin,&cursor,&prior,&current,
                &player,&focus,&x,&y,&valid_buffer,&stable)==21 && obj && stage<3 && mode==0 && flags==4 && tick &&
                stable && valid_buffer && origin && cursor==origin+36+2*tick && current==word){
            int changed=shared->stage!=stage+1 || shared->tick!=tick;
            atomic_fetch_add(&shared->serial,1);shared->stage=stage+1;shared->tick=tick;
            if(changed)shared->upper_ns=after;
            atomic_fetch_add(&shared->serial,1);
        }
        fprintf(out,"%" PRId64 " %" PRId64 " %s",before,after,line);fflush(out);
        struct timespec nap={0,850000};
        nanosleep(&nap,NULL);
    }
    close(requests[1]); fclose(out);fclose(from);
    return 0;
}
