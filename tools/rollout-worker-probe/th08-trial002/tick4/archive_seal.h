/* The shell asks for this handshake only after capturing the original Game Over screen. */
#include <errno.h>
#include <string.h>
#include <sys/wait.h>

static int finish_archive(int descriptor,FILE *from,pid_t child,const unsigned counts[3]){
    char request[96], expected[96], reply[128];
    int n=snprintf(request,sizeof(request),"s %u %u %u\n",counts[0],counts[1],counts[2]);
    snprintf(expected,sizeof(expected),"SEALED %u %u %u",counts[0],counts[1],counts[2]);
    if(write(descriptor,request,(size_t)n)!=n)return 9;
    close(descriptor);
    if(!fgets(reply,sizeof(reply),from))return 9;
    reply[strcspn(reply,"\r\n")]=0;
    fclose(from);
    int status;
    while(waitpid(child,&status,0)<0){if(errno!=EINTR)return 10;}
    if(strcmp(expected,reply) || !WIFEXITED(status) || WEXITSTATUS(status)!=0)return 11;
    FILE *result=fopen("/capture/native-archive-bracket-status.tmp","wx");
    if(!result)return 12;
    fprintf(result,"reader_exit\tstage1_words\tstage2_words\tstage3_words\tack_ns\n0\t%u\t%u\t%u\t%" PRId64 "\n",
            counts[0],counts[1],counts[2],stamp());
    if(fclose(result) || rename("/capture/native-archive-bracket-status.tmp","/capture/native-archive-bracket-status.tsv"))return 12;
    return 0;
}
