#ifndef TOUHOU_NATIVE_LOCATOR_H
#define TOUHOU_NATIVE_LOCATOR_H
#include <stdint.h>
#include "locator_params.h"

/* Spatial CNN operating on original observations. No native state is read here. */
static int fa_locator_index(const uint8_t *frame,const uint8_t *previous,int first){
    static float input[2][112][96],stem[12][56][48];
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
        int i=96*y+x,d=(int)frame[i]-(int)previous[i];
        input[0][y][x]=frame[i]/255.f;
        input[1][y][x]=first?0.f:(d<0?-d:d)/255.f;
    }
    for(int c=0;c<12;c++)for(int y=0;y<56;y++)for(int x=0;x<48;x++){
        float v=loc_stem_b[c];
        for(int ic=0;ic<2;ic++)for(int ky=0;ky<7;ky++){
            int iy=2*y+ky-3;if((unsigned)iy>=112)continue;
            const float *weights=loc_stem_w+((c*2+ic)*7+ky)*7;
            for(int kx=0;kx<7;kx++){
                int ix=2*x+kx-3;if((unsigned)ix<96)v+=input[ic][iy][ix]*weights[kx];
            }
        }
        stem[c][y][x]=v>0?v:0;
    }
    float best=-1e30f;int index=0;
    for(int y=0;y<56;y++)for(int x=0;x<48;x++){
        float score=loc_out_b[0];
        for(int c=0;c<12;c++){
            float v=loc_spatial_b[c];
            for(int ky=0;ky<5;ky++){
                int iy=y+2*ky-4;if((unsigned)iy>=56)continue;
                for(int kx=0;kx<5;kx++){
                    int ix=x+2*kx-4;
                    if((unsigned)ix<48)v+=stem[c][iy][ix]*loc_spatial_w[(c*5+ky)*5+kx];
                }
            }
            if(v>0)score+=v*loc_out_w[c];
        }
        if(score>best){best=score;index=y*48+x;}
    }
    return index;
}
static void fa_locator_features(const uint8_t *frame,const uint8_t *prior,int age,float *features){
    static int last_x,last_y;
    int index=fa_locator_index(frame,prior,age==0),x=(index%48)*2,y=(index/48)*2;
    features[0]=x/48.f-1.f;features[1]=y/112.f;
    features[2]=age?fmaxf(-1.f,fminf(1.f,(x-last_x)/8.f)):0.f;
    features[3]=age?fmaxf(-1.f,fminf(1.f,(y-last_y)/8.f)):0.f;
    last_x=x;last_y=y;
}
#endif
