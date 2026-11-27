/* Benchmark byte-identical frame transport over the same local WSL pipe as the TH08 engineering actor. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern size_t ZSTD_compressBound(size_t);
extern size_t ZSTD_compress(void *,size_t,const void *,size_t,int);
extern unsigned ZSTD_isError(size_t);
static uint64_t ns(void){struct timespec t;clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
int main(int argc,char **argv){
    if(argc!=4)return 2;
    FILE *src=fopen(argv[1],"rb");if(!src)return 3;
    size_t n=strtoul(argv[2],NULL,10),frames=24;
    if(n==0 || n>640*480*3)return 4;
    unsigned char *buf=malloc(n*frames),*dst=malloc(ZSTD_compressBound(n));
    if(!buf||!dst||fread(buf,n,frames,src)!=frames)return 5;
    fclose(src);
    int compress=strcmp(argv[3],"zstd")==0;
    for(;;){
        int id=fgetc(stdin);if(id==EOF||id>=24)break;
        void *raw=buf+(size_t)id*n;void *output=raw;size_t size=n;
        uint64_t start=ns();
        if(compress){size=ZSTD_compress(dst,ZSTD_compressBound(n),raw,n,1);if(ZSTD_isError(size))return 6;output=dst;}
        uint64_t elapsed=ns()-start;
        printf("%zu %llu\n",size,(unsigned long long)elapsed);
        if(fwrite(output,1,size,stdout)!=size||fflush(stdout))return 7;
    }
    free(dst);free(buf);return 0;
}
