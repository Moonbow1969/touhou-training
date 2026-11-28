/* Read-only camera and controller observer for the independently distributed original 0.02. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t tick,stage,keys,prev,cursor,origin,flags,mode; } State;
typedef struct { uint32_t tick,stage,player_state,focus;float x,y; } Position;
typedef struct { DWORD pid; HWND window; } WindowSearch;
static int read_data(HANDLE p,uintptr_t at,void *data,size_t size) {
    SIZE_T n=0;
    return ReadProcessMemory(p,(void *)at,data,size,&n)&&n==size;
}
static int state(HANDLE p,uintptr_t base,State *s){
    uint32_t obj=0,info=0;uint16_t keys,prev;
    memset(s,0,sizeof(*s));
    if(!read_data(p,base+0x14a9340,&obj,4)||!obj) return 0;
    if(!read_data(p,base+0x120c760,&s->stage,4)||s->stage>2) return 0;
    s->stage++;
    if(!read_data(p,obj,&s->tick,4)||!read_data(p,obj+0x10,&s->mode,4)||s->mode!=1) return 0;
    if(!read_data(p,obj+8,&info,4)||!info||!read_data(p,obj+0x50,&s->cursor,4)) return 0;
    if(!read_data(p,info+0x20+4*(s->stage-1),&s->origin,4))return 0;
    if(!read_data(p,base+0x120c9bc,&keys,2)||!read_data(p,base+0x120c9c4,&prev,2))return 0;
    s->keys=keys;s->prev=prev;
    return read_data(p,base+0x120c54c,&s->flags,4)&&(s->flags&4)&&!(s->flags&512);
}
static BOOL CALLBACK check_window(HWND w,LPARAM data) {
    WindowSearch *s=(WindowSearch*)data;DWORD pid=0;RECT r;
    GetWindowThreadProcessId(w,&pid);
    if(pid==s->pid&&IsWindowVisible(w)&&GetClientRect(w,&r)&&r.right==640&&r.bottom==480){s->window=w;return FALSE;}
    return TRUE;
}
static DWORD game_pid(void){
    HANDLE h=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    PROCESSENTRY32 e={0};e.dwSize=sizeof(e);DWORD pid=0;
    if(Process32First(h,&e))do{if(!_stricmp(e.szExeFile,"th08tr.exe")){pid=e.th32ProcessID;break;}}while(Process32Next(h,&e));
    CloseHandle(h);return pid;
}
static int position(HANDLE p,uintptr_t base,const State *s,Position *xy){
    uint8_t player_state[4];
    xy->tick=s->tick;xy->stage=s->stage;
    if(!read_data(p,base+0x13cc810,player_state,4)||
       !read_data(p,base+0x13ccac4,&xy->x,8))return 0;
    xy->player_state=player_state[0];xy->focus=player_state[3];return 1;
}
static void crop(const unsigned char *rgb,unsigned char *out){
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
        unsigned vertical=0;
        for(int u=0;u<4;u++){
            unsigned horizontal=0;
            const unsigned char *p=rgb+4*((16+4*y+u)*640+32+4*x);
            for(int v=0;v<4;v++,p+=4)
                horizontal+=(19595u*p[2]+38470u*p[1]+7471u*p[0]+32768u)>>16;
            vertical+=(horizontal+2)/4;
        }
        out[y*96+x]=(unsigned char)((vertical+2)/4);
    }
}
/* Both crops use the current native position, so recentering does not introduce motion. */
static void player_patch(const unsigned char *bgra,unsigned char *out,int player_x,int player_y){
    for(int y=0;y<64;y++)for(int x=0;x<64;x++){
        int field_x=player_x+x-32,field_y=player_y+y-32;
        unsigned char *pixel=out+3*(64*y+x);
        if(field_x<0||field_x>=384||field_y<0||field_y>=448){
            memset(pixel,0,3);
            continue;
        }
        const unsigned char *source=bgra+4*(640*(16+field_y)+32+field_x);
        pixel[0]=source[2];pixel[1]=source[1];pixel[2]=source[0];
    }
}

int main(int argc,char **argv){
    if(argc!=4&&argc!=6)return 1;
    unsigned wanted=atoi(argv[1]),limit=atoi(argv[2]);
    unsigned patch_limit=argc==6?strtoul(argv[4],NULL,10):0;
    unsigned patch_stride=argc==6?strtoul(argv[5],NULL,10):1;
    if(argc==6&&(!patch_limit||patch_limit>4096||!patch_stride))return 1;
    char root[200];snprintf(root,sizeof(root),"Z:/capture/%s",argv[3]);CreateDirectoryA(root,NULL);
    char path[256];snprintf(path,sizeof(path),"%s/frames.bin",root);
    FILE *out=fopen(path,"wbx");if(!out)return 2;
    snprintf(path,sizeof(path),"%s/observer.tsv",root);FILE *log=fopen(path,"w");if(!log)return 2;
    snprintf(path,sizeof(path),"%s/positions.bin",root);FILE *positions=fopen(path,"wbx");if(!positions)return 2;
    FILE *patches=NULL;
    unsigned char *previous_bgra=NULL;
    if(patch_limit){
        snprintf(path,sizeof(path),"%s/motion-patches.bin",root);patches=fopen(path,"wbx");
        previous_bgra=malloc(640*480*4);
        if(!patches||!previous_bgra)return 2;
    }
    DWORD pid=game_pid();fprintf(log,"game\t%lu\n",pid);fflush(log);if(!pid)return 3;
    HANDLE modules=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,pid);
    MODULEENTRY32 m={0};m.dwSize=sizeof(m);if(!Module32First(modules,&m))return 4;
    CloseHandle(modules);uintptr_t base=(uintptr_t)m.modBaseAddr;
    HANDLE game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|PROCESS_VM_READ,FALSE,pid);if(!game)return 5;
    WindowSearch window={pid,NULL};EnumWindows(check_window,(LPARAM)&window);
    fprintf(log,"window_search\t%lu\t%lx\n",pid,(unsigned long)window.window);fflush(log);
    if(!window.window)return 6;
    HDC desktop=GetDC(NULL),mem=CreateCompatibleDC(desktop);BITMAPINFO bm={{0}};
    bm.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);bm.bmiHeader.biWidth=640;bm.bmiHeader.biHeight=-480;
    bm.bmiHeader.biPlanes=1;bm.bmiHeader.biBitCount=32;bm.bmiHeader.biCompression=BI_RGB;
    unsigned char *rgb=NULL,small[10752];
    HBITMAP bitmap=CreateDIBSection(desktop,&bm,DIB_RGB_COLORS,(void**)&rgb,NULL,0);
    if(!bitmap||!rgb)return 7;HGDIOBJ before=SelectObject(mem,bitmap);
    fprintf(log,"pid\tbase\tgame_window\n%lu\t%lx\t%lx\n",pid,(unsigned long)base,(unsigned long)window.window);fflush(log);
    unsigned accepted=0,racing=0,skipped=0,patch_count=0;int last=-1;
    DWORD started=GetTickCount(),lastGood=started;
    while(GetTickCount()-started<1800000 && GetTickCount()-lastGood<(accepted?18000:1300000)){
        State a,b;Position xy,after;
        if(!state(game,base,&a)||a.stage!=wanted){if(accepted&&a.stage!=wanted&&a.mode==1)break;Sleep(18);continue;}
        if((int)a.tick==last||a.tick%4||a.tick<4){Sleep(1);continue;}
        if(!position(game,base,&a,&xy))break;
        POINT pos={0,0};if(!ClientToScreen(window.window,&pos))break;
        if(!BitBlt(mem,0,0,640,480,desktop,pos.x,pos.y,SRCCOPY)){Sleep(1);continue;}
        GdiFlush();
        if(!state(game,base,&b)||memcmp(&a,&b,sizeof(a))||
           !position(game,base,&b,&after)||memcmp(&xy,&after,sizeof(xy))){racing++;continue;}
        if(a.cursor!=a.origin+0x24+2*a.tick){skipped++;Sleep(1);continue;}
        if(accepted&&(int)a.tick<last)break;
        crop(rgb,small);
        if(accepted%200==0){
            snprintf(path,sizeof(path),"%s/rgb-%06u.bgra",root,a.tick);
            FILE *snap=fopen(path,"wb");if(!snap)return 8;
            fwrite(rgb,4,640*480,snap);fclose(snap);
        }
        fwrite(&a,sizeof(a),1,out);fwrite(small,1,sizeof(small),out);
        fwrite(&xy,sizeof(xy),1,positions);
        if(patches&&patch_count<patch_limit){
            int valid_position=xy.x>=0&&xy.x<384&&xy.y>=0&&xy.y<448;
            if(last>=0&&a.tick==(uint32_t)last+4&&valid_position&&(xy.player_state==0||xy.player_state==3)&&accepted%patch_stride==0){
                unsigned char pair[2*64*64*3];
                int x=(int)(xy.x+.5f),y=(int)(xy.y+.5f);
                player_patch(previous_bgra,pair,x,y);
                player_patch(rgb,pair+64*64*3,x,y);
                if(fwrite(&a.tick,4,1,patches)!=1||fwrite(pair,sizeof(pair),1,patches)!=1)return 10;
                patch_count++;
            }
            memcpy(previous_bgra,rgb,640*480*4);
        }
        last=(int)a.tick;lastGood=GetTickCount();accepted++;
        if(accepted%100==0){fflush(out);fflush(positions);fprintf(log,"rows\t%u\ttick\t%u\tracing\t%u\n",accepted,a.tick,racing);fflush(log);}
        if(accepted>=limit)break;
    }
    fprintf(log,"sealed\t%u\tlast_tick\t%d\tracing\t%u\tcursor_mismatch\t%u\n",accepted,last,racing,skipped);
    fclose(out);fclose(positions);fclose(log);SelectObject(mem,before);DeleteObject(bitmap);DeleteDC(mem);ReleaseDC(NULL,desktop);
    if(patches){fclose(patches);free(previous_bgra);}
    CloseHandle(game);return accepted?0:9;
}
