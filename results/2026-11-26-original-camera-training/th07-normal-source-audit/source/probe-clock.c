/* Read-only clock qualification on the pinned, isolated original TH07 0.11 image. */
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define START 0x48f000u
#define END 0x1356000u
static uint32_t mem[(END-START)/4], prev[(END-START)/4], init[(END-START)/4];
static unsigned char active[(END-START)/4];
static int gamefd(void){
    DIR *dir=opendir("/proc");struct dirent *e;int fd=-1;
    while ((e=readdir(dir))){if (e->d_name[0]<'1'||e->d_name[0]>'9')continue;
        char p[300],n[80];snprintf(p,sizeof(p),"/proc/%s/comm",e->d_name);FILE *f=fopen(p,"r");if(!f)continue;
        if(fgets(n,sizeof(n),f)&&strstr(n,"th07.exe")){snprintf(p,sizeof(p),"/proc/%s/mem",e->d_name);fd=open(p,O_RDONLY);}
        fclose(f);if(fd>=0)break;
    }closedir(dir);return fd;
}
int main(int argc,char**argv){
    int fd=gamefd();if(fd<0){perror("opening original");return 3;}
    if(argc==2){unsigned addr=strtoul(argv[1],NULL,16);unsigned v;if(pread(fd,&v,4,addr)!=4)return 6;printf("%u\n",v);return 0;}
    unsigned adr[2048],sz[2048],num=0;
    FILE *a=fopen("/capture/static-addresses.tsv","r");
    while(num<2048&&fscanf(a,"%x %u",adr+num,sz+num)==2)num++;fclose(a);
    FILE *v=fopen("/capture/word-candidates.bin","wb"),*c=fopen("/capture/clock-candidates.csv","w");
    for(int t=0;t<36;t++){
        if(pread(fd,mem,sizeof(mem),START)!=(ssize_t)sizeof(mem)){perror("original section");return 4;}
        uint32_t values[2048];
        for(unsigned j=0;j<num;j++){values[j]=0;memcpy(&values[j],(unsigned char*)mem+adr[j]-START,sz[j]);}
        fwrite(values,4,num,v);
        for(unsigned n=0;n<sizeof(mem)/4;n++){
            if(!t){init[n]=mem[n];active[n]=mem[n]<8000;}
            else if(active[n]&&(mem[n]<prev[n]+2||mem[n]>prev[n]+36))active[n]=0;
        }
        if(t==35)for(unsigned n=0;n<sizeof(mem)/4;n++)if(active[n]){
            fprintf(c,"%x,%u,%u\n",START+4*n,init[n],mem[n]);
        }
        memcpy(prev,mem,sizeof(mem));fflush(v);usleep(180000);
    }
    fclose(c);fclose(v);close(fd);return 0;
}
