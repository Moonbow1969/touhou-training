FROM debian:bookworm-slim

RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcc-mingw-w64-i686-posix \
       wine wine32:i386 xvfb xdotool libgl1:i386 libgl1-mesa-dri:i386 \
    && rm -rf /var/lib/apt/lists/*

ENV WINEARCH=win32 WINEPREFIX=/tmp/th14-probe-wine WINEDEBUG=-all LIBGL_ALWAYS_SOFTWARE=1
WORKDIR /probe
COPY probe.c run.sh ./
RUN i686-w64-mingw32-gcc-posix -Wall -Wextra -Werror -O2 probe.c -o probe.exe -luser32 -ldinput8 -ldxguid -ld3d9 \
    && chmod +x run.sh
ENTRYPOINT ["/probe/run.sh"]
