/* One controller and one native observer, sharing an explicit idle acknowledgement. */
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
typedef struct { _Atomic unsigned request, idle; } QuietReads;
static QuietReads *quiet_bus;
static int quiet_open(int create) {
    int fd=open("/capture/observer-quiescence",O_RDWR|(create?(O_CREAT|O_EXCL):0),0644);
    if(fd<0)return 0;
    if(create && ftruncate(fd,sizeof(QuietReads))){close(fd);return 0;}
    quiet_bus=mmap(NULL,sizeof(QuietReads),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
    return quiet_bus!=MAP_FAILED;
}
static int64_t quiet_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (int64_t)t.tv_sec*1000000000+t.tv_nsec;}
static void quiet_nap(void){struct timespec t={0,90000};nanosleep(&t,NULL);}
/* Call only in the observer when all ReadProcessMemory responses have returned. */
static inline int quiet_between_reads(void){
    unsigned epoch=atomic_load(&quiet_bus->request);
    if(!(epoch&1))return 1;
    atomic_store(&quiet_bus->idle,epoch);
    int64_t deadline=quiet_now()+1200000000;
    while(atomic_load(&quiet_bus->request)==epoch){
        if(quiet_now()>=deadline)return 0;
        quiet_nap();
    }
    return 1;
}
static inline void quiet_release(void){
    if(quiet_bus!=NULL && quiet_bus!=MAP_FAILED && (atomic_load(&quiet_bus->request)&1))atomic_fetch_add(&quiet_bus->request,1);
}
static inline int quiet_request(void){
    if(atomic_load(&quiet_bus->request)&1)return 0;
    unsigned epoch=atomic_fetch_add(&quiet_bus->request,1)+1;
    int64_t deadline=quiet_now()+30000000;
    while(atomic_load(&quiet_bus->idle)!=epoch){
        if(quiet_now()>=deadline){quiet_release();return 0;}
        quiet_nap();
    }
    return 1;
}
