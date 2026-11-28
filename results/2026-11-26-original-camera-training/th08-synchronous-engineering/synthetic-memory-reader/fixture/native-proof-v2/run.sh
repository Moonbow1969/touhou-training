set -eu
cd /capture
i686-w64-mingw32-gcc-posix -O2 -std=c11 sync_memory_wine_fixture.c -o proofsrc.exe
cp proofsrc.exe proofread.exe
export WINEARCH=win32 WINEPREFIX=/tmp/wine WINEDEBUG=-all DISPLAY=:97
Xvfb :97 -screen 0 640x480x24 >/tmp/x.log 2>&1 &
wine proofsrc.exe target >game.log 2>&1 &
for i in $(seq 1 300);do if test -s target.txt; then break;fi;sleep .2;done
test -s target.txt
sourcepid=$(for file in /proc/[0-9]*/comm; do if grep -qx proofsrc.exe "$file" 2>/dev/null; then basename "$(dirname "$file")"; fi; done)
test -n "$sourcepid"
./monitor "$sourcepid" no_reader >no-reader.csv
wine proofread.exe reader >reader.log 2>&1 &
for i in $(seq 1 80);do if test -s reader-ready; then break;fi;sleep .1;done
test -s reader-ready
./monitor "$sourcepid" wine_read_process_memory >with-reader.csv
