/* Capture only the original 640x480 client's playfield while the clock is frozen. */
static int emit_camera(HWND window) {
    static HDC display, memory;
    static unsigned char *bits;
    static char output[112*96*2+1];
    RECT client; POINT origin={32,16};
    if(!GetClientRect(window,&client)||client.right!=640||client.bottom!=480||!ClientToScreen(window,&origin))return 0;
    if(!display){
        BITMAPINFO info={0}; info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
        info.bmiHeader.biWidth=384;info.bmiHeader.biHeight=-448;info.bmiHeader.biPlanes=1;
        info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
        display=GetDC(NULL);memory=CreateCompatibleDC(display);
        HBITMAP image=CreateDIBSection(display,&info,DIB_RGB_COLORS,(void **)&bits,NULL,0);
        if(!display||!memory||!image||!bits||!SelectObject(memory,image))return 0;
    }
    if(!BitBlt(memory,0,0,384,448,display,origin.x,origin.y,SRCCOPY))return 0;
    GdiFlush();
    static const char digits[]="0123456789abcdef";
    for(int y=0;y<112;y++)for(int x=0;x<96;x++){
        unsigned sum=0;
        for(int dy=0;dy<4;dy++)for(int dx=0;dx<4;dx++){
            const unsigned char *pixel=bits+4*((y*4+dy)*384+x*4+dx);
            sum+=77*pixel[2]+150*pixel[1]+29*pixel[0];
        }
        unsigned value=(sum+2048)/4096;int offset=2*(y*96+x);
        output[offset]=digits[value>>4];output[offset+1]=digits[value&15];
    }
    printf("\"camera\":\"%s\",",output);return 1;
}
