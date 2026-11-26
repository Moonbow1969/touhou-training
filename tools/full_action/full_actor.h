#ifndef TOUHOU_FULL_ACTOR_H
#define TOUHOU_FULL_ACTOR_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef FA_LEARNED_LOCATOR
#include "native_locator.h"
#define FA_MAGIC "THVIS001"
#else
#define FA_MAGIC "THFULL01"
#endif

/* Parameter order follows torch.nn.Linear (out, in). One process per live game. */
enum { FA_INPUT=110, FA_HIDDEN=96, FA_OUTPUT=16, FA_PIXELS=112*96 };
static float fa_w1[96][110], fa_b1[96], fa_w2[96][96], fa_b2[96], fa_out[16][96], fa_bias[16];
static float actor_features[110], actor_logp[4], actor_probs[15], actor_value;
static uint8_t fa_previous[FA_PIXELS];
static int fa_action_history[2][4], fa_age;
static uint64_t fa_state;
static void actor_reset(uint64_t seed){
    fa_state=seed; fa_age=0;
    memset(fa_action_history,0,sizeof fa_action_history);
    memset(fa_previous,0,sizeof fa_previous);
}
static int actor_init(const char *path,uint64_t seed){
    FILE *fp=fopen(path,"rb"); char magic[8]; int ok;
    if(!fp)return 1;
    ok=fread(magic,1,8,fp)==8 && memcmp(magic,FA_MAGIC,8)==0;
#define FA_LOAD(a) (fread(a,1,sizeof(a),fp)==sizeof(a))
    ok=ok && FA_LOAD(fa_w1) && FA_LOAD(fa_b1) && FA_LOAD(fa_w2) && FA_LOAD(fa_b2)
          && FA_LOAD(fa_out) && FA_LOAD(fa_bias) && fgetc(fp)==EOF;
#undef FA_LOAD
    fclose(fp); actor_reset(seed); return !ok;
}
static double fa_uniform(void){
    uint64_t z=(fa_state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return ((z^(z>>31))>>11)*0x1.0p-53;
}
static void actor_step(const uint8_t *frame,int out[4]){
    memset(actor_features,0,sizeof actor_features);
    for(int cell=0;cell<42;cell++){
        int sum=0,change=0,top=(cell/6)*16,left=(cell%6)*16;
        for(int y=top;y<top+16;y++)for(int x=left;x<left+16;x++){
            int i=y*96+x,d=(int)frame[i]-(int)fa_previous[i];
            sum+=frame[i]; change+=d<0?-d:d;
        }
        actor_features[cell]=(float)sum/65280.f;
        actor_features[42+cell]=fa_age?(float)change/65280.f:0.f;
    }
#ifdef FA_LEARNED_LOCATOR
    fa_locator_features(frame,fa_previous,fa_age,actor_features);
#endif
    memcpy(fa_previous,frame,FA_PIXELS);
    for(int lag=0;lag<2 && lag<fa_age;lag++){
        int p=84+lag*12;
        actor_features[p+fa_action_history[lag][0]]=1.f;
        for(int h=1;h<4;h++) actor_features[p+8+h]=fa_action_history[lag][h];
    }
    actor_features[108]=fminf(fa_age/120.f,1.f);
    actor_features[109]=fminf(fa_age/1800.f,1.f);
    float hidden[96],latent[96],logits[16];
    for(int j=0;j<96;j++){float v=fa_b1[j];for(int i=0;i<110;i++)v+=fa_w1[j][i]*actor_features[i];hidden[j]=tanhf(v);}
    for(int j=0;j<96;j++){float v=fa_b2[j];for(int i=0;i<96;i++)v+=fa_w2[j][i]*hidden[i];latent[j]=tanhf(v);}
    for(int j=0;j<16;j++){float v=fa_bias[j];for(int i=0;i<96;i++)v+=fa_out[j][i]*latent[i];logits[j]=v;}
    actor_value=logits[15];
    int off=0;
    for(int h=0;h<4;h++){
        int width=h?2:9;float top=logits[off];double sum=0.;
        for(int a=1;a<width;a++)if(logits[off+a]>top)top=logits[off+a];
        for(int a=0;a<width;a++)sum+=exp((double)(logits[off+a]-top));
        double u=fa_uniform(),cumulative=0.;out[h]=width-1;
        for(int a=0;a<width;a++){
            actor_probs[off+a]=(float)(exp((double)(logits[off+a]-top))/sum);
            cumulative+=actor_probs[off+a];
            if(u<cumulative){out[h]=a;u=2.;}
        }
        actor_logp[h]=logf(actor_probs[off+out[h]]);off+=width;
    }
    memcpy(fa_action_history[1],fa_action_history[0],sizeof fa_action_history[0]);
    memcpy(fa_action_history[0],out,sizeof fa_action_history[0]);fa_age++;
}
#endif
