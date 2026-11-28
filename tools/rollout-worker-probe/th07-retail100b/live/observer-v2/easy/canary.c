#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <stdlib.h>
static volatile sig_atomic_t done;
static void finish(int unused) { (void)unused; done=1; }
static int64_t ns(struct timespec t){return (int64_t)t.tv_sec*1000000000+t.tv_nsec;}
int main(void){
    struct timespec start,now;
    signal(SIGTERM,finish);
    clock_gettime(CLOCK_MONOTONIC,&start);
    int64_t deadline=ns(start)+1000000;
    int n=0,late1=0,miss=0;
    int64_t peak=0,peakTime=0;
    puts("sample_end_ns,samples,over_1ms,missed_wakes,max_late_ns,peak_ns");
    while(!done){
        struct timespec next={deadline/1000000000,deadline%1000000000};
        if(clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&next,0)==EINTR)continue;
        clock_gettime(CLOCK_MONOTONIC,&now);
        int64_t now_ns=ns(now),delay=now_ns-deadline;
        if(delay>peak){peak=delay;peakTime=now_ns;}
        if(delay>=1000000)late1++;
        int skipped=delay>0?(int)(delay/1000000):0;
        miss+=skipped; n++;
        deadline+=(skipped+1)*1000000LL;
        if(n+miss>=1000 || done){
            printf("%lld,%d,%d,%d,%lld,%lld\n",(long long)now_ns,n,late1,miss,(long long)peak,(long long)peakTime);
            fflush(stdout); n=late1=miss=0;peak=peakTime=0;
        }
    }
    return 0;
}
