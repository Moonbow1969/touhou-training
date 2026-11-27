#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
/* P6 pixels are the stock 640x480 client. Three masks are the frozen native rank glyphs. */
int main(int argc,char **argv){
    if(argc!=3)return 2;
    FILE *img=fopen(argv[1],"rb"),*glyphs=fopen(argv[2],"rb");
    if(!img||!glyphs)return 2;
    char line[128],magic[4]; int w=0,h=0,max=0;
    if(!fgets(line,sizeof line,img)||sscanf(line,"%3s",magic)!=1||magic[0]!='P'||magic[1]!='6')return 2;
    do{if(!fgets(line,sizeof line,img))return 2;}while(line[0]=='#');
    if(sscanf(line,"%d %d",&w,&h)!=2||!fgets(line,sizeof line,img)||sscanf(line,"%d",&max)!=1||w!=640||h!=480||max!=255)return 2;
    static uint8_t pixels[480][640][3], masks[3][27][260];
    if(fread(pixels,1,sizeof pixels,img)!=sizeof pixels||fread(masks,1,sizeof masks,glyphs)!=sizeof masks)return 2;
    for(int rank=0;rank<3;rank++){
        unsigned overlap=0,n=0;
        for(int y=0;y<27;y++)for(int x=0;x<260;x++){
            uint8_t *p=pixels[200+y][190+x];int observed=p[0]<55&&p[1]<55&&p[2]<55,expected=masks[rank][y][x];
            n+=observed+expected;overlap+=observed&&expected;
        }
        if(n && 2.0*overlap/n>=.98){printf("%d\n",rank+1);return 0;}
    }
    return 3;
}
