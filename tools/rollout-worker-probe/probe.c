#define COBJMACROS
#define DIRECTINPUT_VERSION 0x0800
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <d3d9.h>
#include <dinput.h>

struct GameState {
    LONG ticks;
    LONG keys;
};

static struct GameState state;
static LPDIRECTINPUTDEVICE8A keyboard;
static BOOL previous_z;

static LRESULT CALLBACK window_events(HWND window, UINT message, WPARAM key, LPARAM data) {
    if (message == WM_TIMER) {
        state.ticks++;
        BYTE keys[256];
        if (FAILED(IDirectInputDevice8_GetDeviceState(keyboard, sizeof(keys), keys))) {
            IDirectInputDevice8_Acquire(keyboard);
        } else {
            BOOL current_z = (keys[DIK_Z] & 0x80) != 0;
            state.keys += current_z && !previous_z;
            previous_z = current_z;
        }
        return 0;
    }
    return DefWindowProcA(window, message, key, data);
}

static BOOL initialize_keyboard(HWND window, HINSTANCE instance) {
    LPDIRECTINPUT8A input;
    if (FAILED(DirectInput8Create(instance, DIRECTINPUT_VERSION, &IID_IDirectInput8A, (void **)&input, NULL)))
        return FALSE;
    HRESULT created = IDirectInput8_CreateDevice(input, &GUID_SysKeyboard, &keyboard, NULL);
    IDirectInput8_Release(input);
    return SUCCEEDED(created) && SUCCEEDED(IDirectInputDevice8_SetDataFormat(keyboard, &c_dfDIKeyboard))
           && SUCCEEDED(IDirectInputDevice8_SetCooperativeLevel(keyboard, window,
                                                              DISCL_FOREGROUND | DISCL_NONEXCLUSIVE));
}

static BOOL render_one_frame(HWND window) {
    IDirect3D9 *graphics = Direct3DCreate9(D3D_SDK_VERSION);
    if (!graphics)
        return FALSE;
    D3DPRESENT_PARAMETERS present = {0};
    present.BackBufferFormat = D3DFMT_UNKNOWN;
    present.SwapEffect = D3DSWAPEFFECT_DISCARD;
    present.hDeviceWindow = window;
    present.Windowed = TRUE;
    present.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    IDirect3DDevice9 *device;
    HRESULT created = IDirect3D9_CreateDevice(graphics, D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window,
                                              D3DCREATE_SOFTWARE_VERTEXPROCESSING, &present, &device);
    BOOL rendered = FALSE;
    if (SUCCEEDED(created)) {
        HRESULT cleared = IDirect3DDevice9_Clear(device, 0, NULL, D3DCLEAR_TARGET, 0xff306098, 1., 0);
        rendered = SUCCEEDED(cleared) && SUCCEEDED(IDirect3DDevice9_Present(device, NULL, NULL, NULL, NULL));
        IDirect3DDevice9_Release(device);
    }
    IDirect3D9_Release(graphics);
    return rendered;
}

static int run_target(const char *mapping_name) {
    HANDLE mapping = OpenFileMappingA(FILE_MAP_WRITE, FALSE, mapping_name);
    if (!mapping)
        return 2;
    volatile uintptr_t *address = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, sizeof(*address));
    if (!address)
        return 3;

    HINSTANCE instance = GetModuleHandleA(NULL);
    WNDCLASSA kind = {0};
    kind.lpfnWndProc = window_events;
    kind.hInstance = instance;
    kind.lpszClassName = "Th14Probe";
    if (!RegisterClassA(&kind))
        return 4;
    HWND window = CreateWindowA(kind.lpszClassName, "TH14 environment probe", WS_OVERLAPPEDWINDOW,
                                100, 100, 646, 509, NULL, NULL, instance, NULL);
    if (!window || !initialize_keyboard(window, instance) || !render_one_frame(window)
        || !SetTimer(window, 1, 16, NULL)) {
        fputs("DirectInput or Direct3D 9 initialization failed\n", stderr);
        return 5;
    }
    *address = (uintptr_t)&state;
    ShowWindow(window, SW_SHOW);

    MSG event;
    while (GetMessageA(&event, NULL, 0, 0) > 0)
        DispatchMessageA(&event);
    return 0;
}

static BOOL read_state(HANDLE target, uintptr_t address, struct GameState *value) {
    SIZE_T count = 0;
    return ReadProcessMemory(target, (const void *)address, value, sizeof(*value), &count)
           && count == sizeof(*value);
}

static int test_target(PROCESS_INFORMATION process, volatile uintptr_t *address) {
    struct GameState running = {0};
    for (int retry = 0; retry < 240; retry++) {
        Sleep(50);
        if (*address && read_state(process.hProcess, *address, &running) && running.keys == 1)
            break;
    }
    if (!running.ticks || running.keys != 1) {
        fprintf(stderr, "Initial memory or keyboard validation failed, ticks=%ld keys=%ld\n",
                running.ticks, running.keys);
        return 6;
    }
    if (!DebugActiveProcess(process.dwProcessId)) {
        fprintf(stderr, "DebugActiveProcess failed: %lu\n", GetLastError());
        return 7;
    }
    DebugSetProcessKillOnExit(FALSE);
    Sleep(100);
    struct GameState frozen;
    struct GameState later;
    struct GameState resumed;
    BOOL read_frozen = read_state(process.hProcess, *address, &frozen);
    Sleep(150);
    BOOL read_later = read_state(process.hProcess, *address, &later);
    BOOL detached = DebugActiveProcessStop(process.dwProcessId);
    Sleep(150);
    BOOL read_resumed = read_state(process.hProcess, *address, &resumed);
    if (!(read_frozen && read_later && detached && read_resumed))
        return 8;
    printf("{\"directinput_keys\":%ld,\"d3d9_present\":true,\"frozen_ticks\":%ld,"
           "\"later_ticks\":%ld,\"resumed_ticks\":%ld}\n",
           resumed.keys, frozen.ticks, later.ticks, resumed.ticks);
    return frozen.ticks == later.ticks && resumed.ticks > later.ticks && resumed.keys == 1 ? 0 : 9;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--target") == 0)
        return run_target(argv[2]);

    char name[64];
    snprintf(name, sizeof(name), "Local\\TH14Probe-%lu", GetCurrentProcessId());
    HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(uintptr_t), name);
    if (!mapping)
        return 10;
    volatile uintptr_t *address = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(*address));
    if (!address)
        return 11;

    STARTUPINFOA startup = {0};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process;
    char command[96];
    snprintf(command, sizeof(command), "probe.exe --target %s", name);
    if (!CreateProcessA(NULL, command, NULL, NULL, FALSE, 0, NULL, NULL, &startup, &process))
        return 12;

    int result = test_target(process, address);
    TerminateProcess(process.hProcess, 0);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    UnmapViewOfFile((const void *)address);
    CloseHandle(mapping);
    return result;
}
