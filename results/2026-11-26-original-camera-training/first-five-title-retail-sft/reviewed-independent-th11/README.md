# Prospective TH11 player test

| Actor | Stages | Players | All directions | Movement starts | Turns |
|---|---:|---:|---:|---:|---:|
| four | 3 | 3 | 4,131/8,425 | 48/717 | 123/926 |
| five | 3 | 3 | 4,267/8,425 | 71/717 | 135/926 |
| five-complete | 3 | 3 | 4,445/8,425 | 37/717 | 151/926 |
| five-reviewed | 3 | 3 | 4,450/8,425 | 34/717 | 133/926 |

Results include qualified members of the four separately registered external players. Each model predicts human actions on the same original screens. [Stage and event metrics](held.json)

One Hard and two Lunatic players qualified. The fourth player's two Hard captures agreed with native input and inventory, but both fell below the 80% independently registered direction-tracking threshold. The denominator above excludes that player. Adding all ten TH11 training stages improved overall direction accuracy over the four-title control on each of the three admitted players. Correct counts for horikaze were 1,141 → 1,169 of 2,329, zjm 1,277 → 1,344 of 2,538, and north 1,713 → 1,937 of 3,558. North accounts for 224 of the 319 additional correct decisions. The newest actor also correctly predicted fewer movement starts (34/717) than the four-title control (48/717).

These players used Bomb on three distinct onsets spanning six action windows. No model selected Bomb as its highest-probability choice on any of the 8,425 windows. For the ten-stage TH11 actor, the mean predicted Bomb probability on those six windows was 2.35%. All three onsets belong to north. [Independent original camera archive and checksum](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th11-retail-independent-four-held-cameras-receipt.json)
