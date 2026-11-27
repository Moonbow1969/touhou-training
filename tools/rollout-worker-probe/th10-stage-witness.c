/* Engineering-only read-only original-state controller for the Stage 1 -> 2 sensor witness. */
#define main unused_step_entry
#include "th10-trial-step.c"
#undef main

static float clamp_position(float value,float lower,float upper) {
    return fminf(upper,fmaxf(lower,value));
}

static float score_direction(const struct Th10State *s,int count,int dx,int dy,int last_dx,int last_dy,
                             float *imminent_clearance) {
    float speed=(dx&&dy)?1.4142f:2.f;
    float path_cost=0.f,min_clearance=1e5f;
    for(int tick=1;tick<=32;tick++){
        float x=clamp_position(s->xy[0]+dx*speed*tick,-182.f,182.f);
        float y=clamp_position(s->xy[1]+dy*speed*tick,20.f,429.f);
        float travel_x=x-s->xy[0],travel_y=y-s->xy[1];
        float nearest=1e5f;
        for(int i=0;i<count&&i<96;i++){
            float bx=objects[i].relative[0]+objects[i].velocity[0]*tick-travel_x;
            float by=objects[i].relative[1]+objects[i].velocity[1]*tick-travel_y;
            float distance=bx*bx+by*by;
            nearest=fminf(nearest,distance);
        }
        if(tick<=9)min_clearance=fminf(min_clearance,nearest);
        float gap=sqrtf(nearest);
        float danger=fmaxf(0.f,17.f-gap);
        path_cost+=danger*danger/(1.f+tick*.055f);
    }
    *imminent_clearance=sqrtf(min_clearance);
    float nx=clamp_position(s->xy[0]+dx*speed*13.f,-182.f,182.f);
    float ny=clamp_position(s->xy[1]+dy*speed*13.f,20.f,429.f);
    float goal=.004f*fabsf(nx)+.006f*fabsf(ny-370.f);
    return -path_cost-goal-.06f*(dx!=last_dx||dy!=last_dy)-.12f*(dx||dy);
}

int main(void) {
    DWORD pid=0;
    HANDLE game=th10_process(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,&pid);
    if(!game)return 2;
    FILE *log=fopen("Z:\\capture\\diagnostic.csv","w");
    if(!log)return 3;
    fprintf(log,"ms,utc_ms,legacy_stage,candidate_stage,difficulty,spares,power_units,x,y,bullets,requested_mask,native_input,clearance\n");
    uint64_t beginning=GetTickCount64(),last_bomb=0,transition=0;
    int last_dx=0,last_dy=0,ever_stage1=0;
    while(GetTickCount64()-beginning<8*60*1000){
        uint64_t now=GetTickCount64(),t=now-beginning;
        struct Th10State s;uint32_t power=0,input=0,candidate_stage=0;
        if(!th10_read_state(game,&s)||!th10_read(game,TH10_SCORE+4,&power,sizeof(power))||
            !th10_read(game,CURRENT_NATIVE_INPUT,&input,sizeof(input))||
            !th10_read(game,0x472b40,&candidate_stage,sizeof(candidate_stage))){Sleep(12);continue;}
        if(s.difficulty!=0||s.lives<0)break;
        if(candidate_stage==1)ever_stage1=1;
        if(ever_stage1&&candidate_stage==2&&!transition){
            transition=t;FILE *flag=fopen("Z:\\capture\\stage2-marker","w");
            if(flag){fprintf(flag,"%llu\n",(unsigned long long)t);fclose(flag);}
        }
        if(transition&&t-transition>22000)break;
        int count=projectiles(game,s.xy);
        if(count<0){Sleep(12);continue;}
        float best=-1e15f,chosen_clear=1e5f;int move_x=0,move_y=0;
        for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
            float clearance;
            float score=score_direction(&s,count,x,y,last_dx,last_dy,&clearance);
            if(score>best){best=score;move_x=x;move_y=y;chosen_clear=clearance;}
        }
        /* Spend available original Power during the dense late Stage 1 boss patterns.
           This engineering fixture needs a positive stage readback, not resource efficiency. */
        if(power>=20&&((chosen_clear<23.f&&now-last_bomb>2300)||
                       (t>122000&&count>28&&now-last_bomb>4300)))last_bomb=now;
        int mask=(move_x<0?1:move_x>0?2:0)|(move_y<0?4:move_y>0?8:0)|16;
        if(t%520>25)mask|=32; /* Repeated original Shoot presses also advance conversations. */
        if(last_bomb&&now-last_bomb<70)mask|=64;
        FILETIME ft;ULARGE_INTEGER ft_ticks;GetSystemTimeAsFileTime(&ft);
        ft_ticks.LowPart=ft.dwLowDateTime;ft_ticks.HighPart=ft.dwHighDateTime;
        uint64_t utc_ms=ft_ticks.QuadPart/10000-UINT64_C(11644473600000);
        fprintf(log,"%llu,%llu,%u,%u,%u,%d,%u,%.2f,%.2f,%d,%d,%u,%.2f\n",
                (unsigned long long)t,(unsigned long long)utc_ms,s.stage,candidate_stage,s.difficulty,s.lives,power,s.xy[0],s.xy[1],count,mask,input,chosen_clear);
        fflush(log);printf("%d\n",mask);fflush(stdout);
        last_dx=move_x;last_dy=move_y;Sleep(21);
    }
    printf("0\n");fflush(stdout);fclose(log);CloseHandle(game);
    return 0;
}
