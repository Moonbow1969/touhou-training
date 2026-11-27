#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct Display Display;
typedef struct XImage XImage;
typedef unsigned long Window;

static Display *(*XOpenDisplay)(const char *);
static int (*XCloseDisplay)(Display *);
static Window (*XDefaultRootWindow)(Display *);
static XImage *(*XGetImage)(Display *, Window, int, int, unsigned, unsigned, unsigned long, int);
static unsigned long (*XGetPixel)(XImage *, int, int);
static int (*XDestroyImage)(XImage *);
static int (*XQueryKeymap)(Display *, char[32]);
static int (*XSync)(Display *, int);
static unsigned long (*XStringToKeysym)(const char *);
static unsigned char (*XKeysymToKeycode)(Display *, unsigned long);
static int (*XTestFakeKeyEvent)(Display *, unsigned, int, unsigned long);

#define LOAD(h, f) do { *(void **)(&f) = dlsym(h, #f); if (!f) fail("Missing display symbol: " #f); } while (0)

enum { SCENE_X = 35, SCENE_Y = 38, WIDTH = 384, HEIGHT = 448, SMALL_W = 96, SMALL_H = 112 };

struct Patch {
    int x, y, width, height;
    const char *path;
    unsigned char bytes[8192];
    double norm;
};

static struct Patch rank_title = {121, 44, 139, 22, "/input/ranking-full.gray", {0}, 0};
static struct Patch pause_title = {12, 205, 72, 29, "/input/pause.gray", {0}, 0};
static struct Patch game_over_title = {12, 205, 72, 29, "/input/gameover.gray", {0}, 0};
static struct Patch retry_selected = {34, 345, 137, 25, "/input/retry.gray", {0}, 0};
static struct Patch confirm_title = {203, 249, 150, 36, "/input/confirm.gray", {0}, 0};
static struct Patch hud_title = {460, 5, 72, 19, "/input/hud.gray", {0}, 0};
static struct Patch opening_title = {32, 374, 99, 20, "/input/title.gray", {0}, 0};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void initialize(void) {
#ifdef TH12_MULTI_RANK_HUD
    const char *difficulty = getenv("TH12_DIFFICULTY");
    if (difficulty && !strcmp(difficulty,"2")) hud_title.path = "/input/hud-hard.gray";
    if (difficulty && !strcmp(difficulty,"3")) hud_title.path = "/input/hud-lunatic.gray";
#endif
    void *x = dlopen("libX11.so.6", RTLD_NOW);
    void *t = dlopen("libXtst.so.6", RTLD_NOW);
    if (!x || !t) fail("X11 display libraries are unavailable.");
    LOAD(x, XOpenDisplay); LOAD(x, XCloseDisplay); LOAD(x, XDefaultRootWindow);
    LOAD(x, XGetImage); LOAD(x, XGetPixel); LOAD(x, XDestroyImage); LOAD(x, XSync);
    LOAD(x, XQueryKeymap); LOAD(x, XStringToKeysym); LOAD(x, XKeysymToKeycode); LOAD(t, XTestFakeKeyEvent);
}

static int gray(XImage *image, int x, int y) {
    unsigned long pixel = XGetPixel(image, x, y);
    return (77 * ((pixel >> 16) & 255) + 150 * ((pixel >> 8) & 255) + 29 * (pixel & 255)) >> 8;
}

static int sample(const struct Patch *p, int x, int y) { return p->bytes[y * p->width + x]; }

static int template_edge(const struct Patch *p, int x, int y) {
    return 4 * sample(p, x, y) - sample(p, x-1, y) - sample(p, x+1, y)
        - sample(p, x, y-1) - sample(p, x, y+1);
}

static void load_patch(struct Patch *p) {
    FILE *file = fopen(p->path, "rb");
    size_t expected = (size_t)p->width * p->height;
    if (!file || expected > sizeof p->bytes || fread(p->bytes, 1, expected, file) != expected || fgetc(file) != EOF)
        fail("The private TH12 state template has the wrong shape.");
    fclose(file);
    for (int row = 1; row < p->height-1; ++row) {
        for (int col = 1; col < p->width-1; ++col) {
            double edge = template_edge(p, col, row);
            p->norm += edge * edge;
        }
    }
}

static double match(XImage *image, const struct Patch *p) {
    double dot = 0, square = 0;
    for (int row = 1; row < p->height-1; ++row) {
        for (int col = 1; col < p->width-1; ++col) {
            int x = p->x + col, y = p->y + row;
            double edge = 4 * gray(image,x,y)-gray(image,x-1,y)-gray(image,x+1,y)-gray(image,x,y-1)-gray(image,x,y+1);
            dot += edge * template_edge(p, col, row);
            square += edge * edge;
        }
    }
    return dot / sqrt(fmax(square * p->norm, 1));
}

static const char *screen(XImage *image, int status) {
    double p = match(image, &pause_title), g = match(image, &game_over_title);
    double r = match(image, &retry_selected), c = match(image, &confirm_title), h = match(image, &hud_title);
    const char *label = NULL;
    if (match(image,&rank_title) > .72) label = "score_ranking";
    else if (c > .63) label = "confirm";
    else if (g > .63 && g > p + .12) label = r > .84 ? "gameover_retry" : "gameover";
    else if (p > .63 && p > g + .12) label = r > .84 ? "pause_retry" : "pause";
    else if (h < .4 && match(image, &opening_title) > .82) label = "title";
    else {
        int light = 0;
        for (int y = 8; y < HEIGHT; y += 32)
            for (int x = 8; x < WIDTH; x += 32)
                light += gray(image, x, y) > 38;
        label = light >= 55 && h > .65 ? "play" : "loading";
    }
    if (status) printf("%s %.3f %.3f %.3f %.3f %.3f\n", label, p, g, r, c, h);
    return label;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) fail("Could not read the monotonic capture clock.");
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void wait_until(uint64_t ns) {
    struct timespec clock = {ns / 1000000000ULL, ns % 1000000000ULL};
    int rc;
    do { rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &clock, NULL); } while (rc == EINTR);
    if (rc) fail("Could not sleep until the next observation.");
}

static XImage *image_now(Display *d, Window root) {
    XImage *image = XGetImage(d, root, SCENE_X, SCENE_Y, 608, HEIGHT, ~0UL, 2);
    if (!image) fail("No playfield returned by the private display.");
    return image;
}

static void key(Display *d, unsigned code, int pressed) {
    if (!XTestFakeKeyEvent(d, code, pressed, 0)) fail("Private display rejected an action.");
}

static int resource_hud(XImage *image, int y_offset) {
    int lives = 0, gap = 0;
    for (int icon=0; icon<8; ++icon) {
        int count=0;
        for (int y=91+y_offset; y<102+y_offset; ++y) {
            for (int x=483+13*icon; x<496+13*icon; ++x) {
                unsigned long px = XGetPixel(image,x,y);
                int r=(int)((px>>16)&255), g=(int)((px>>8)&255), b=(int)(px&255);
                int maximum = r>g ? (r>b?r:b) : (g>b?g:b);
                int minimum = r<g ? (r<b?r:b) : (g<b?g:b);
                count += minimum > 185 && maximum-minimum < 38;
            }
        }
        if (count < 20) gap=1;
        else if (gap) return -1;
        else ++lives;
    }
    return lives;
}
