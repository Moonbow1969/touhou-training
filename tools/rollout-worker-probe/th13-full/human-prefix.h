/* Engineering only. Drive ordinary X11 controls from a frozen human replay prefix.
   The policy receives a fresh camera history when the prefix finishes. */
#ifndef TOHO_PREFIX_TARGET
#define TOHO_PREFIX_TARGET 5000
#endif
static int engineering_prefix(Display *display,int fd,int32_t state[21]) {
    enum { target=TOHO_PREFIX_TARGET };
    uint16_t words[target];
    FILE *source=fopen("/capture/lib/human-prefix.u16","rb");
    FILE *log=fopen("prefix.csv","w");
    fprintf(stderr,"prefix opening stage=%d rank=%d character=%d clock=%d spares=%d bombs=%d\n",
            state[3],state[8],state[9],state[11],state[4],state[5]);
    if(!source||!log||fread(words,sizeof(words[0]),target,source)!=target)return 0;
    fclose(source);
    if(state[3]!=1||state[8]!=2||state[9]!=0||state[11]>100||state[11]<0)return 0;
    (void)system("scrot -o prefix-opening.png >/dev/null 2>&1 &");
    fprintf(log,"stage_tick,stage,rank,character,spares,bombs,power,life_fragments,bomb_fragments,score,native_key,human_key_after_tick,requested_next_key,x11_key_after_request,monotonic_ns\n");
    int previous_tick=-1,old=0;
    const int64_t start=now_ns();
    while(now_ns()-start<400000000000LL) {
        if(!native(fd,state)||state[2]!=2||state[3]!=1||state[8]!=2||state[4]<0||state[7]!=0)break;
        const int tick=state[11];
        if(tick<previous_tick && previous_tick>60)break;
        if(tick==previous_tick){hold_ms(1);continue;}
        previous_tick=tick;
        int after=tick>0&&tick<=target?(int)words[tick-1]&251:0;
        int next=tick<target?(int)words[tick]&251:0;
        if(old!=next)control(display,old,next);
        old=next;
        fprintf(log,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld\n",
            tick,state[3],state[8],state[9],state[4],state[5],state[18],state[16],state[17],state[6],
            state[10]&251,after,next,held(display),(long long)now_ns());
        if(tick%60==0)fflush(log);
        if(tick>=target) {
            fclose(log);
            (void)system("scrot -o handoff-original.png >/dev/null 2>&1 &");
            return tick<=target+6;
        }
    }
    control(display,old,0);fclose(log);return 0;
}
