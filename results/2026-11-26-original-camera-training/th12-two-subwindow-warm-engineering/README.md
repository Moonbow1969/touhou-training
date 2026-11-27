# Native control timing on Steam Deck

Each model received 96 ordinary Hard gameplay images from the original TH12 trial. In two cold starts, the first physical input landed four native ticks late. Four zero-image forwards before opening the game connection reduced the measured first-command delay to zero ticks in both warm starts. Prewarming preserved Torch RNG. Every one of the 95 between-image intervals was four ticks in both warm games.

With two predicted subcommands per image, the original game consumed the requested input on 384/384 ticks. The model changed the command midway 72 times and all 72 changes were observed at both allocated ticks. The four-tick reference matched on 383/384 ticks. These engineering credits used independent seeds and measured input delivery.

[All native clocks and model hashes](prewarm-native-clock-comparison.json) · [Raw four-tick reference](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th12-two-subwindow-warm-engineering-th12-prewarmed-four-tick-original-clock.zip) · [Raw two-command stream](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th12-two-subwindow-warm-engineering-th12-prewarmed-two-subwindows-original-clock.zip)
