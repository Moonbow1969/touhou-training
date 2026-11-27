# TH08 original Hard and Lunatic visual control

The unmodified Imperishable Night trial v0.03 ran in an isolated Wine virtual desktop on Gamer. A two-frame, four-head visual PPO previously trained on TH10, TH11, TH12, TH13 and TH17 controlled every gameplay key. It had yet to receive any TH08 online training. Two earlier technical attempts stopped on bright spell effects with two lives still left. We preserved the false positives, added a structural check for the continue prompt and required zero spare life stars. [Rejected attempts](rejected-brightness-only-pilots.json) · [Original trial provenance](trial-provenance.json) · [Publisher download page](https://www16.big.or.jp/~zun/html/th08dl.html)

| Original difficulty | Fresh seed | Physical policy actions | Opening to continue screen | Final spares | Final Bombs | Actual directions used |
|---|---:|---:|---:|---:|---:|---:|
| Hard | 1620801 | 816 | 93.45 s | 0 | 2 | 9/9 |
| Lunatic | 1620802 | 666 | 77.91 s | 0 | 2 | 9/9 |

Each credit started with two spare lives and three Bombs. Both reached the first original continue prompt. The original difficulty glyph and HUD are visible in each video. Native X-server keyboard state was checked after every model action, and all three stock losses can be accounted for, including the final death. The table contains one engineering credit per difficulty. [Hard full video](1620801/original-observations.mp4) · [Hard physical inputs](1620801/receipt.json) · [Lunatic full video](1620802/original-observations.mp4) · [Lunatic physical inputs](1620802/receipt.json)

A separate registered study collects fresh TH08 stocks from this fixed five-title actor. Its first update uses four new credits per difficulty, then compares the original and adapted actors on eight new paired seeds per difficulty. A joint six-title branch combines 32 new TH08 credits with the 160 original five-title third-generation training credits. [Training and test registrations](../six-game-online/protocol.json)
