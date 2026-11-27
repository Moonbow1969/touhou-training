#ifndef TH10_CAMERA_ACTOR_H
#define TH10_CAMERA_ACTOR_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

/* These 110 coarse features are diagnostics. The TorchScript model receives the raw camera. */
#define FA_CAMERA_ACTOR 1
static float actor_features[110],actor_camera_logits[15],actor_probs[15],actor_logp[4],actor_value;
#ifdef FA_CAMERA_EXPLICIT_PREVIOUS_MOVE
static uint8_t actor_camera_previous_move;
#endif
static int camera_fd=-1;
static uint64_t camera_rng;
static uint8_t camera_previous[112*96];
static int camera_age,camera_actions[2][4];
static double camera_uniform(void){
    uint64_t z=(camera_rng+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return ((z^(z>>31))>>11)*0x1.0p-53;
}
static int camera_connect(void){
    struct sockaddr_un address={0};address.sun_family=AF_UNIX;
    struct timeval deadline={.tv_sec=6};
    const char *path=getenv("TOHO_CAMERA_SOCKET");if(!path)path="/camera/actor.sock";
    if(strlen(path)>=sizeof address.sun_path)return 1;
    strcpy(address.sun_path,path);
    camera_fd=socket(AF_UNIX,SOCK_STREAM,0);
    if(camera_fd<0)return 1;
    setsockopt(camera_fd,SOL_SOCKET,SO_RCVTIMEO,&deadline,sizeof deadline);
    setsockopt(camera_fd,SOL_SOCKET,SO_SNDTIMEO,&deadline,sizeof deadline);
    if(connect(camera_fd,(struct sockaddr *)&address,sizeof address)<0)return 2;
    const char *expected=getenv("TOHO_CAMERA_SHA");char actual[64];
    if(!expected || strlen(expected)!=64 || recv(camera_fd,actual,64,MSG_WAITALL)!=64 || memcmp(actual,expected,64))return 3;
#ifdef FA_CAMERA_EXPLICIT_PREVIOUS_MOVE
    char mode[2];
    if(recv(camera_fd,mode,sizeof mode,MSG_WAITALL)!=(ssize_t)sizeof mode || memcmp(mode,"H1",2))return 4;
#endif
    return 0;
}
static void actor_reset(uint64_t seed){
    if(camera_fd>=0)close(camera_fd);
    camera_fd=-1;camera_rng=seed;camera_age=0;
    memset(camera_actions,0,sizeof camera_actions);memset(camera_previous,0,sizeof camera_previous);
    if(camera_connect()!=0){if(camera_fd>=0)close(camera_fd);camera_fd=-1;}
}
static int actor_init(const char *model,uint64_t seed){
    (void)model;actor_reset(seed);return camera_fd<0;
}
static void camera_diagnostics(const uint8_t *frame){
    memset(actor_features,0,sizeof actor_features);
    for(int cell=0;cell<42;cell++){
        int sum=0,change=0,top=(cell/6)*16,left=(cell%6)*16;
        for(int y=top;y<top+16;y++)for(int x=left;x<left+16;x++){
            int i=y*96+x;sum+=frame[i];change+=abs((int)frame[i]-(int)camera_previous[i]);
        }
        actor_features[cell]=(float)sum/65280.f;
        actor_features[42+cell]=camera_age?(float)change/65280.f:0.f;
    }
    for(int lag=0;lag<2 && lag<camera_age;lag++){
        int p=84+lag*12;actor_features[p+camera_actions[lag][0]]=1;
        for(int h=1;h<4;h++)actor_features[p+8+h]=camera_actions[lag][h];
    }
    actor_features[108]=fminf(camera_age/120.f,1);actor_features[109]=fminf(camera_age/1800.f,1);
    memcpy(camera_previous,frame,sizeof camera_previous);
}
static void actor_step(const uint8_t *frame,int out[4]){
    camera_diagnostics(frame);
#ifdef FA_CAMERA_EXPLICIT_PREVIOUS_MOVE
    actor_camera_previous_move=camera_age?(uint8_t)camera_actions[0][0]:4;
    if(send(camera_fd,&actor_camera_previous_move,1,MSG_NOSIGNAL)!=1){out[0]=-1;return;}
#endif
    size_t sent=0,n=112u*96u;
    while(sent<n){
        ssize_t wrote=send(camera_fd,frame+sent,n-sent,MSG_NOSIGNAL);
        if(wrote<=0){out[0]=-1;return;}sent+=(size_t)wrote;
    }
    float response[16];
    if(recv(camera_fd,response,sizeof response,MSG_WAITALL)!=(ssize_t)sizeof response){out[0]=-1;return;}
    memcpy(actor_camera_logits,response,sizeof actor_camera_logits);actor_value=response[15];
    if(!isfinite(actor_value)){out[0]=-1;return;}
    int off=0;
    for(int h=0;h<4;h++){
        int width=h?2:9;float top=actor_camera_logits[off];double sum=0.;
        for(int a=0;a<width;a++){
            if(!isfinite(actor_camera_logits[off+a])){out[0]=-1;return;}
            if(actor_camera_logits[off+a]>top)top=actor_camera_logits[off+a];
        }
        for(int a=0;a<width;a++)sum+=exp((double)(actor_camera_logits[off+a]-top));
        float prob[9];double draw=camera_uniform(),cumulative=0.;out[h]=width-1;
        for(int a=0;a<width;a++){
            prob[a]=(float)(exp((double)(actor_camera_logits[off+a]-top))/sum);actor_probs[off+a]=prob[a];
            cumulative+=prob[a];if(draw<cumulative){out[h]=a;draw=2.;}
        }
        actor_logp[h]=logf(prob[out[h]]);off+=width;
    }
    memcpy(camera_actions[1],camera_actions[0],sizeof camera_actions[0]);
    memcpy(camera_actions[0],out,sizeof camera_actions[0]);camera_age++;
}
#endif
