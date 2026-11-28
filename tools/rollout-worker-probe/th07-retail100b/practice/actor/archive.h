/* Single producer, single disk writer. Main thread performs no synchronous
   disk writes during ordinary play. A full ring or failed writer fails closed. */
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#define CAPACITY 4096
typedef struct { uint32_t sequence, lane, length; char line[256]; } Evidence;
static Evidence queue[CAPACITY];
static atomic_uint produced, consumed, finished, fault;
static pthread_t writer;
static uint32_t crc_table[256];
static uint32_t record_crc(const Evidence *r){
    uint32_t x=UINT32_MAX;
    const unsigned char *bytes=(const unsigned char*)r;
    for(size_t i=0;i<12+r->length;i++)x=(x>>8)^crc_table[(x^bytes[i])&255];
    return ~x;
}
static void *drain_archive(void *unused){
    (void)unused;
    FILE *paths[2]={fopen("/capture/native-input.csv","wx"),fopen("/capture/applied.csv","wx")};
    FILE *journal=fopen("/capture/events.bin","wbx");
    if(!paths[0]||!paths[1]||!journal){atomic_store(&fault,1);return NULL;}
    fputs("ns,tick,word,previous_word,stage,stageclock,flags,retry,continues,rank,shot,lives,bombs,misses,bombs_used,score,record_slot,previous_slot,event_index\n",paths[0]);
    fputs("sequence,camera_tick,applied_tick,requested_word,apply_ns,frozen_tick,freeze_start_ns,resume_ns\n",paths[1]);
    uint32_t at=0;
    for(;;){
        if(at==atomic_load_explicit(&produced,memory_order_acquire)){
            if(atomic_load(&finished) && at==atomic_load(&produced))break;
            usleep(500);continue;
        }
        const Evidence *r=&queue[at%CAPACITY];
        uint32_t crc=record_crc(r);
        if(r->sequence!=at||r->lane>1||fwrite(r,1,12+r->length,journal)!=12+r->length||
           fwrite(&crc,4,1,journal)!=1||fwrite(r->line,1,r->length,paths[r->lane])!=r->length){
            atomic_store(&fault,1);break;
        }
        atomic_store_explicit(&consumed,++at,memory_order_release);
    }
    FILE *all[3]={paths[0],paths[1],journal};
    for(unsigned i=0;i<3;i++)if(fflush(all[i])||fsync(fileno(all[i]))||fclose(all[i]))atomic_store(&fault,1);
    return NULL;
}
static void archive_open(void){
    for(uint32_t i=0;i<256;i++){
        uint32_t v=i;
        for(unsigned j=0;j<8;j++)v=(v>>1)^((v&1)?0xedb88320u:0);
        crc_table[i]=v;
    }
    if(pthread_create(&writer,NULL,drain_archive,NULL))exit(60);
}
static void append_event(uint32_t lane,const char *format,...){
    const uint32_t at=atomic_load(&produced);
    if(atomic_load(&fault)||at-atomic_load_explicit(&consumed,memory_order_acquire)>=CAPACITY)exit(61);
    Evidence *r=&queue[at%CAPACITY];
    r->sequence=at;r->lane=lane;
    va_list args;va_start(args,format);int n=vsnprintf(r->line,sizeof(r->line),format,args);va_end(args);
    if(n<1||(unsigned)n>=sizeof(r->line))exit(62);
    r->length=n;atomic_store_explicit(&produced,at+1,memory_order_release);
}
static void archive_seal(void){
    atomic_store(&finished,1);
    if(pthread_join(writer,NULL)||atomic_load(&fault)||atomic_load(&consumed)!=atomic_load(&produced))exit(63);
    FILE *seal=fopen("/capture/archive.sealed","wx");if(!seal)exit(64);
    if(fprintf(seal,"%u\n",atomic_load(&consumed))<0||fflush(seal)||fsync(fileno(seal))||fclose(seal))exit(65);
}
