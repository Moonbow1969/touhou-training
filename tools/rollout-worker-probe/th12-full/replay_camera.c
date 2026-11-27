#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DWORD game_pid;
static HWND game_window;
static HANDLE game;
static uint32_t read32(uintptr_t address) {
    uint32_t value=UINT32_MAX; SIZE_T count=0;
    if(!ReadProcessMemory(game,(const void *)address,&value,sizeof value,&count)||count!=sizeof value)return UINT32_MAX;
    return value;
}
static BOOL CALLBACK find_window(HWND window,LPARAM user) {
    (void)user;
    RECT rect; DWORD owner;
    GetWindowThreadProcessId(window,&owner);
    if(owner==game_pid && IsWindowVisible(window) && GetClientRect(window,&rect) && rect.right==640 && rect.bottom==480){
        game_window=window;return FALSE;
    }
    return TRUE;
}
static int connect_game(void) {
    PROCESSENTRY32 pe={0}; pe.dwSize=sizeof pe;
    HANDLE ss=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
    if(Process32First(ss,&pe))do{
        if(!_stricmp(pe.szExeFile,"th12.exe")){ if(game_pid)return 0;game_pid=pe.th32ProcessID; }
    }while(Process32Next(ss,&pe)); CloseHandle(ss);
    if(!game_pid)return 0;
    game=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,FALSE,game_pid);
    EnumWindows(find_window,0);
    return game && game_window;
}
struct state { uint32_t manager, mode, tick, stage, global_stage, keys, lives, bombs; };
static struct state state_now(void) {
    uint32_t m=read32(0x4b21d4);
    struct state s={m,read32(m+0x10),read32(m+0x100),read32(m+0x108),
                    read32(0x4aea38),read32(0x4d2670),read32(0x4aea20),read32(0x4aea28)};
    return s;
}
int main(int argc,char **argv) {
    unsigned wanted=argc>1?(unsigned)atoi(argv[1]):1; if(wanted<1||wanted>3)return 7;
    if(!connect_game())return 2;
    FILE *meta=fopen("Z:/capture/human-telemetry.tsv","wb");
    FILE *cam=fopen("Z:/capture/human-camera.u8","wb");
    if(!meta || !cam)return 3;
    HDC screen=GetDC(game_window), memory=CreateCompatibleDC(screen);
    BITMAPINFO info={0}; info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=384;
    info.bmiHeader.biHeight=-448;info.bmiHeader.biPlanes=1; info.bmiHeader.biBitCount=32;
    info.bmiHeader.biCompression=BI_RGB;
    uint8_t *rgb=NULL; HBITMAP bitmap=CreateDIBSection(screen,&info,DIB_RGB_COLORS,(void **)&rgb,NULL,0);
    if(!rgb || !bitmap)return 4; SelectObject(memory,bitmap);
    uint32_t last=UINT32_MAX,stage=UINT32_MAX;unsigned count=0;DWORD start=GetTickCount();
    fprintf(meta,"tick\tkeys\tstage\tglobal_stage\tlives\tbombs\tmode\tclock_ms\tbracket_ms\n");fflush(meta);
    DWORD last_seen=start;
    while(GetTickCount()-start<660000 && count<7000){
        struct state a=state_now();
        if(count && ((a.mode==1 && a.stage==a.global_stage && a.stage>0 && a.stage!=stage) ||
                     GetTickCount()-last_seen>8000 || (a.stage==stage && a.tick+16<last)))break;
        if(a.mode!=1 || a.tick==UINT32_MAX || a.stage!=wanted || a.stage!=a.global_stage){Sleep(2);continue;}
        if(a.tick%4 || a.tick==last){Sleep(1);continue;}
        DWORD begin=GetTickCount();
        if(!BitBlt(memory,0,0,384,448,screen,32,16,SRCCOPY)){Sleep(1);continue;}
        GdiFlush();
        struct state b=state_now();
        if(a.tick!=b.tick || a.keys!=b.keys || a.stage!=b.stage || a.mode!=b.mode || a.global_stage!=b.global_stage){Sleep(1);continue;}
        uint8_t out[112*96];
        for(int y=0;y<112;y++)for(int x=0;x<96;x++){
            unsigned total=0;
            for(int j=0;j<4;j++)for(int i=0;i<4;i++){
                uint8_t *p=rgb+4*((4*y+j)*384+4*x+i);
                total+=29*p[0]+150*p[1]+77*p[2];
            }
            out[y*96+x]=(uint8_t)((total+2048)/4096);
        }
        if(fwrite(out,sizeof out,1,cam)!=1)return 5;
        fprintf(meta,"%u\t%u\t%u\t%u\t%u\t%u\t%u\t%lu\t%lu\n",a.tick,a.keys,a.stage,a.global_stage,a.lives,a.bombs,a.mode,
                (unsigned long)begin,(unsigned long)(GetTickCount()-begin));
        if(!(count%32)){fflush(cam);fflush(meta);}
        count++;last=a.tick;stage=a.stage;last_seen=GetTickCount();Sleep(1);
    }
    fclose(cam);fclose(meta);DeleteDC(memory);ReleaseDC(game_window,screen);DeleteObject(bitmap);CloseHandle(game);
    return count?0:6;
}
