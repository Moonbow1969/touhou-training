#!/bin/sh
# Make a fresh engineering-only clone of a previously verified TH08 0.02 continuous four-tick host deployment.
set -eu
if [ "$#" -ne 2 ]; then echo 'Usage: prepare_quiescent_operator.sh QUALIFIED_STUDY NEW_STUDY' >&2; exit 2; fi
base=$(realpath "$1")
dest=$2
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cc=${CC:-gcc}
(cd "$base/app" && sha256sum -c tick4-deployment.sha256 >/dev/null && grep -Fx /clock/stream postcredit-ranking.sh >/dev/null && grep -F /clock/bracket game-entry-v2.sh >/dev/null)
test -d "$base/deps"
mkdir "$dest"
mkdir "$dest/app" "$dest/games"
cp -a --reflink=auto "$base/app/." "$dest/app/"
ln -s "$(realpath "$base/deps")" "$dest/deps"
cd "$dest/app"
for name in stream.c bracket.c gate.h archive_seal.h sync_gate.h quiet_reads.h; do cp "$src/$name" "tick4/$name"; done
cp "$src/../../th13-pixels/x11_pixels.h" tick4/x11_pixels.h
"$cc" -std=c11 -O2 -D_GNU_SOURCE -DTH08_SYNCHRONOUS_ENGINEERING -DTH08_QUIET_READS tick4/stream.c -o tick4/stream -l:libX11.so.6 -l:libXtst.so.6
"$cc" -std=c11 -O2 -static -DTH08_EXPLICIT_ARCHIVE_SEAL -DTH08_QUIET_READS tick4/bracket.c -o tick4/bracket
cp tick4/stream assets/stream
cut -d' ' -f3- "$base/app/tick4-deployment.sha256" | xargs sha256sum > tick4-deployment.sha256
sha256sum tick4/sync_gate.h tick4/quiet_reads.h >>tick4-deployment.sha256
sha256sum -c tick4-deployment.sha256 >/dev/null
sha256sum tick4/stream tick4/bracket policy.ts
