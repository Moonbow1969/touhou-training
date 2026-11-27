#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <time.h>
int main(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) return 1;
    return printf("%lld.%09ld\n", (long long)value.tv_sec, value.tv_nsec)<0;
}
