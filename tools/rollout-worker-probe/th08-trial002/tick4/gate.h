/* Read a paired Linux timestamp and original game tick published under a seqlock. */
#include <fcntl.h>
#include <sys/mman.h>
#include <stdatomic.h>
typedef struct { _Atomic unsigned serial,stage,tick; _Atomic int64_t upper_ns; } TickShared;
static TickShared *tick_source;
static FILE *gate_log;
static unsigned observed_stage;
static int poll_tick(void) {
    for(int n=0;n<90;n++){
        unsigned a=atomic_load_explicit(&tick_source->serial,memory_order_acquire);
        unsigned stage=tick_source->stage;
        unsigned t=tick_source->tick;
        int64_t ns=tick_source->upper_ns;
        unsigned b=atomic_load_explicit(&tick_source->serial,memory_order_acquire);
        if(a==b && a && !(a&1) && now_ns()-ns<25000000 && t>0){
            observed_stage=stage;
            return (int)t;
        }
        hold_ms(1);
    }
    return -1;
}
static int wait_tick(int target){
    int current;
    unsigned stage=observed_stage;
    int64_t deadline=now_ns()+250000000;
    do {
        current=poll_tick();
        if(current<0)return -1;
        if(current>=target || observed_stage!=stage)return current;
        hold_ms(1);
    }while(now_ns()<deadline);
    return -1;
}
