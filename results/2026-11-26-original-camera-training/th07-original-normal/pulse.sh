#!/bin/sh
set -eu
cd /capture
press() { xdotool keydown "$1"; sleep .10; xdotool keyup "$1"; }
snap() { scrot -o "$1.png"; printf '%s %s\n' "$(date +%s%N)" "$1" >> pulse.tsv; }
snap shot-menu-before-confirm
press z
sleep 2.8
snap initial-1
sleep .35
snap initial-2
xdotool keydown Shift_L
sleep .25
snap focused-center
xdotool keydown Left
sleep .36
snap focus-left-1
sleep .36
xdotool keyup Left
snap focus-left-2
xdotool keydown Right
sleep .36
snap focus-right-1
sleep .36
xdotool keyup Right
snap focus-right-2
xdotool keydown Up
sleep .42
xdotool keyup Up
snap focused-up
xdotool keyup Shift_L
xdotool keydown z
sleep .24
snap live-shot
xdotool keyup z
press x
sleep .40
snap live-bomb
sleep 1
snap after-bomb
press Escape
sleep .4
snap stock-paused
