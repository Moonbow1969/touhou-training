# One six-game camera policy trained with four TH08 teams

The optimizer used 192 ordinary original-game credits and 100,479 playable decisions on TH08, 10, 11, 12, 13 and 17. Each title supplied 16 Hard and 16 Lunatic games. The TH08 input covered 11,669 Hard and 10,070 Lunatic decisions.

The two six-game fits start from the same five-game actor and critic. They share all 160 original credits from the other games and eight TH08 Border Team credits. The four-team arm replaces 24 Border credits with eight each from Magic, Scarlet and Ghost. All four teams contribute four credits per difficulty. The 24 replacement credits supplied 17,383 decisions on Entropy. The reference used Gamer for its TH08 data, so collection host and physical cadence can also affect the comparison.

Actor SHA-256 is `5009c0b827435657f0f7f1c075750db76c5bdd9de3ba584d6c960470b3f532e3`. The physical test roster has eight matched seeds per team and difficulty, for 64 pairs. Both arms must play each member of a pair on the same Entropy lane. Native game randomness is uncontrolled.

[Checkpoint and epoch metrics](joint-six-four-teams/policy/report.json) · [Deployable policy](joint-six-four-teams/policy/policy.ts) · [Seed, rank and team assignments](team-held-registration.json) · [All optimizer inputs by hash](joint-six-four-teams/optimizer-input/credits.json) · [Original credit outcomes](original-training-results.json)

[24 new optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-multiteam-online-new-team-optimizer-input.zip)
 · [th08-magic-training-evidence](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-multiteam-online-th08-magic-training-evidence.zip)
 · [th08-scarlet-training-evidence](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-multiteam-online-th08-scarlet-training-evidence.zip)
 · [th08-ghost-training-evidence](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-multiteam-online-th08-ghost-training-evidence.zip)
