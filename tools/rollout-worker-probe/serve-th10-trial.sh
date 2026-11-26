#!/bin/bash
set -eu

window=$1
difficulty=${2:-1}
xdotool windowfocus --sync "$window"
keys=(Left Right Up Down Shift z)
held=0

apply_keys() {
    local next=$1 idx bit
    local down=() up=()
    for idx in "${!keys[@]}"; do
        bit=$((1 << idx))
        if (( (held ^ next) & bit )); then
            if (( next & bit )); then
                down+=("${keys[idx]}")
            else
                up+=("${keys[idx]}")
            fi
        fi
    done
    if (( ${#up[@]} )); then xdotool keyup "${up[@]}"; fi
    if (( ${#down[@]} )); then xdotool keydown "${down[@]}"; fi
    held=$next
}
trap 'apply_keys 0' EXIT

start_sensor() {
    coproc SENSOR { wine /tmp/th10-step.exe "$difficulty"; }
    sensor_pid=$SENSOR_PID
    exec {reader}<&"${SENSOR[0]}" {writer}>&"${SENSOR[1]}"
    read -r -t 8 -u "$reader" state
    printf '%s\n' "$state"
}

release_sensor() {
    printf 'Q\n' >&"$writer"
    exec {writer}>&- {reader}<&-
    wait "$sensor_pid"
}

retry_game() {
    apply_keys 0
    release_sensor
    xdotool keydown Escape; sleep .15; xdotool keyup Escape; sleep .65
    xdotool keydown r; sleep .15; xdotool keyup r; sleep 2
    # The calling environment independently checks fresh score, stock and spawn.
    start_sensor
}

start_sensor
while read -r op mask; do
    op=${op%$'\r'}
    mask=${mask%$'\r'}
    if test "$op" = Q; then break; fi
    if [[ "$op" == R && -z "$mask" ]]; then
        retry_game
        continue
    fi
    if [[ "$op" != S || ! "$mask" =~ ^[0-9]{2}$ ]]; then
        printf 'Invalid TH10 control line.\n' >&2
        exit 2
    fi
    (( mask >= 32 && mask <= 63 && (mask & 3) != 3 && (mask & 12) != 12 )) || exit 2
    apply_keys "$mask"
    printf 'S %s\n' "$mask" >&"$writer"
    read -r -t 8 -u "$reader" state
    printf '%s\n' "$state"
done
apply_keys 0
release_sensor
