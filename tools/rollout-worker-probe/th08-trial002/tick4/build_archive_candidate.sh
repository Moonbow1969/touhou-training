#!/bin/sh
# Build only. Invoke with a fresh output directory outside the running credits.
set -eu
out=${1:?Usage: build_archive_candidate.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cc=${CC:-i686-w64-mingw32-gcc-posix}
for item in 'reader_async.c clock.exe' 'archive_writer_fixture.c archive-fixture.exe'; do
    set -- $item
    "$cc" -std=c11 -O2 -Wall -Wextra -Werror -Wl,--no-insert-timestamp "$src/$1" -o "$out/$2"
done
sha256sum "$src/reader_async.c" "$src/archive_writer.h" "$src/archive_writer_fixture.c" "$out/clock.exe" "$out/archive-fixture.exe"
