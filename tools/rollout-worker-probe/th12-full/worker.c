#include "hud.h"
#ifdef HUMAN_ACTOR
#ifdef CAMERA_ACTOR
#include "camera_actor.h"
#else
#include "full_actor.h"
#endif
#endif
#define KEYS 7
static unsigned codes[KEYS];
static unsigned mask_for(const int a[4]) {
    const int dx=a[0]%3-1,dy=a[0]/3-1;
    return ((dx<0)<<0)|((dx>0)<<1)|((dy<0)<<2)|((dy>0)<<3)|(a[1]<<4)|(a[2]<<5)|(a[3]<<6);
}
static void set_keys(Display *d,unsigned old,unsigned next){
    for(int k=0;k<KEYS;k++) if((old&(1u<<k)) && !(next&(1u<<k))) key(d,codes[k],0);
    for(int k=0;k<KEYS;k++) if(!(old&(1u<<k)) && (next&(1u<<k))) key(d,codes[k],1);
    XSync(d,0);
}
static unsigned read_keys(Display *d){
    char map[32]; XQueryKeymap(d,map); unsigned mask=0;
    for(int k=0;k<KEYS;k++)if((unsigned char)map[codes[k]/8]&(1u<<(codes[k]%8)))mask|=1u<<k;
    return mask;
}
static void playfield(XImage *im,unsigned char *out){
    for(int y=0;y<SMALL_H;y++)for(int x=0;x<SMALL_W;x++){
        unsigned total=0;
        for(int j=0;j<4;j++)for(int i=0;i<4;i++){
            unsigned long v=XGetPixel(im,4*x+i,4*y+j);
            total+=77*((v>>16)&255)+150*((v>>8)&255)+29*(v&255);
        }
        out[y*SMALL_W+x]=(unsigned char)((total+2048)/4096);
    }
}
static void save_terminal(XImage *im){
    /* Keep the exact XImage that triggered termination. Menu inputs can change the
       window before a later independent screenshot. The crop begins at (32,16)
       within the native 640x480 window; excluded borders are stored as black. */
    FILE *fp=fopen("terminal-at-control.ppm","wbx");
    if(!fp)fail("Cannot create original terminal observation.");
    fprintf(fp,"P6\n640 480\n255\n");
    unsigned char row[640*3];
    for(int y=0;y<480;y++){
        memset(row,0,sizeof row);
        if(y>=16 && y<16+HEIGHT)for(int x=32;x<640;x++){
            unsigned long pixel=XGetPixel(im,x-32,y-16);
            row[3*x]=(unsigned char)(pixel>>16);
            row[3*x+1]=(unsigned char)(pixel>>8);
            row[3*x+2]=(unsigned char)pixel;
        }
        if(fwrite(row,1,sizeof row,fp)!=sizeof row)fail("Terminal image write failed.");
    }
    if(fclose(fp))fail("Terminal image close failed.");
}
int main(int argc,char **argv){
    if(argc==2 && !strcmp(argv[1],"clock")){
        uint64_t stamp=now_ns();
        printf("%llu.%09llu\n",(unsigned long long)(stamp/1000000000ULL),
               (unsigned long long)(stamp%1000000000ULL));
        return 0;
    }
    if(argc==2 && (!strcmp(argv[1],"status") || !strcmp(argv[1],"hud"))){
        initialize();load_patch(&rank_title);load_patch(&pause_title);load_patch(&game_over_title);
        load_patch(&retry_selected);load_patch(&confirm_title);load_patch(&hud_title);load_patch(&opening_title);
        Display *d=XOpenDisplay(NULL);if(!d)fail("Cannot open private display.");
        XImage *im=image_now(d,XDefaultRootWindow(d));
        if(!strcmp(argv[1],"status"))printf("%s\n",screen(im,0));
        else printf("lives %d\nbombs %d\n",resource_hud(im,0),resource_hud(im,24));
        XDestroyImage(im);XCloseDisplay(d);
        return 0;
    }
    if(argc!=5) fail("Expected model seed hold_ms qualify|train|held");
    const int qual=!strcmp(argv[4],"qualify");
    const int training=!strcmp(argv[4],"train"),held=!strcmp(argv[4],"held");
#ifdef CAMERA_ACTOR
    const int save_frames=1;
#else
    const int save_frames=qual||training;
#endif
    unsigned ms=(unsigned)atoi(argv[3]); uint64_t seed=strtoull(argv[2],NULL,10);
    if(!seed || ms<60 || ms>300 || (!qual&&!training&&!held)) fail("Invalid collector parameters.");
#ifdef HUMAN_ACTOR
    if(!qual && actor_init(argv[1],seed)!=0)fail("Cannot load signed human actor.");
    if(!qual) actor_reset(seed);
#else
    if(!qual)fail("Policy weights not linked into this qualification binary.");
#endif
    initialize();
    load_patch(&rank_title);load_patch(&pause_title);load_patch(&game_over_title);load_patch(&retry_selected);
    load_patch(&confirm_title);load_patch(&hud_title);load_patch(&opening_title);
    Display *d=XOpenDisplay(NULL); if(!d)fail("Cannot open X display.");
    Window root=XDefaultRootWindow(d);
    const char *names[KEYS]={"Left","Right","Up","Down","Shift_L","z","x"};
    for(int k=0;k<KEYS;k++){codes[k]=XKeysymToKeycode(d,XStringToKeysym(names[k]));if(!codes[k])fail("Missing action key.");}
    FILE *rows=fopen("decisions.tsv","wx");
    FILE *f=qual?NULL:fopen("features.f32","wbx");
    FILE *pixels=save_frames?fopen("observations.u8","wbx"):NULL;
    if(!rows||(!qual&&!f)||(save_frames&&!pixels))fail("Cannot create fresh trajectory.");
#ifdef CAMERA_ACTOR
    FILE *camera_logits=qual?NULL:fopen("camera-logits.f32","wbx");
    if(!qual && !camera_logits)fail("Cannot create camera policy trace.");
#endif
    fprintf(rows,"index\tonset_ns\tobserved_ns\tmove\tfocus\tshoot\tbomb\tkeys\tverified\tspares_before\tbombs_before\tspares\tbombs\tstatus\tlogp_move\tlogp_focus\tlogp_shoot\tlogp_bomb\tvalue\n");
    const int test[][4]={{1,0,1,0},{7,0,0,0},{3,1,1,0},{5,1,0,0},{0,0,1,1},{2,1,0,0},{6,0,1,0},{8,1,1,0},{4,0,0,0}};
    unsigned current=0; int index=0; uint64_t started=now_ns(),last_live=started;
    XImage *im=image_now(d,root); const char *status=screen(im,0);
    if(strcmp(status,"play")||resource_hud(im,0)!=2||resource_hud(im,24)!=2)fail("Initial stock mismatch.");
    const char *outcome="technical_censor";
    while(now_ns()-started<UINT64_C(1200000000000)){
        if(!strcmp(status,"gameover")||!strcmp(status,"gameover_retry")||!strcmp(status,"score_ranking")){outcome="game_over";break;}
        if(strcmp(status,"play")){
            set_keys(d,current,0);current=0;
            if(strcmp(status,"loading")||now_ns()-last_live>UINT64_C(35000000000))break;
            XDestroyImage(im);wait_until(now_ns()+UINT64_C(50000000));
            im=image_now(d,root);status=screen(im,0);continue;
        }
        last_live=now_ns();
        int a[4];float lp[4]={0},value=0;
        unsigned char frame[SMALL_W*SMALL_H];playfield(im,frame);
        if(qual){
            if(index==36){outcome="input_qualified";break;}
            memcpy(a,test[index/4],sizeof a);
        }
#ifdef HUMAN_ACTOR
        else {
            actor_step(frame,a);
            if(a[0]<0 || a[0]>8 || (unsigned)a[1]>1 || (unsigned)a[2]>1 || (unsigned)a[3]>1){
                set_keys(d,current,0);
                fail("Policy service did not return a valid full action.");
            }
            memcpy(lp,actor_logp,sizeof lp);value=actor_value;
            if(fwrite(actor_features,sizeof(float),110,f)!=110)fail("Feature write failed.");
#ifdef CAMERA_ACTOR
            if(fwrite(actor_camera_logits,sizeof(float),15,camera_logits)!=15)fail("Camera logits write failed.");
#endif
        }
#endif
        if(save_frames&&fwrite(frame,1,sizeof frame,pixels)!=sizeof frame)fail("Pixel write failed.");
        int before_l=resource_hud(im,0),before_b=resource_hud(im,24);
        XDestroyImage(im);
        const unsigned next=mask_for(a);set_keys(d,current,next);current=next;
        unsigned verified=read_keys(d);
        if(verified!=next)fail("X keyboard map disagrees with sampled action.");
        uint64_t onset=now_ns(),due=onset+(uint64_t)ms*1000000;
        wait_until(due);im=image_now(d,root);status=screen(im,0);
        unsigned long long stamp=now_ns();
        fprintf(rows,"%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%u\t%u\t%d\t%d\t%d\t%d\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\n",index,(unsigned long long)onset,stamp,a[0],a[1],a[2],a[3],next,verified,before_l,before_b,resource_hud(im,0),resource_hud(im,24),status,lp[0],lp[1],lp[2],lp[3],value);
        ++index;
        if(index%32==0)fflush(rows);
    }
    set_keys(d,current,0);
    if(!strcmp(outcome,"game_over")){
        save_terminal(im);
        FILE *state=fopen("terminal-state.txt","wx");
        if(!state || fprintf(state,"%s\n",status)<0 || fclose(state))fail("Cannot write terminal menu state.");
    }
    XDestroyImage(im);XCloseDisplay(d);
    if(fclose(rows)||(f&&fclose(f))||(pixels&&fclose(pixels)))fail("Trajectory close failed.");
#ifdef CAMERA_ACTOR
    if(camera_logits && fclose(camera_logits))fail("Camera logits close failed.");
#endif
    printf("%s %d %.6f\n",outcome,index,(now_ns()-started)/1e9);
    return !strcmp(outcome,"technical_censor")?3:0;
}
