# TH07 retail Normal Reimu A source census

The user's Gamer PC contains the **original Japanese 1.00b executable** and game data. The executable is 650,752 bytes. Its SHA-256 exactly matches the independently published target. A separately supplied trial executable lives in the research workspace. The first original compatibility run is tracked separately. [Target hash and source](build.json) · [ZUN version history](https://www16.big.or.jp/~zun/html/th07_update.html)

| Embedded recording patch | Normal Reimu A recordings with Stage 1 + 2, by archive URL | Named uploader handles | Handles matching local executable size and recorded checksum |
|---|---:|---:|---:|
| 1.00 | 1 | 1 | 0 |
| 1.00a | 1 | 1 | 0 |
| 1.00b | 449 | 273 | 271 |

The source sweep covers 556 previously catalogued retail URLs from Gensokyo, Silent Selene and Royalflare. The native rank, shot and stage table qualify 451 URLs. For 1.00b, removing unspecified uploader names leaves 442 entries. Of these, 438 report both the local binary size and the original executable checksum. Reuploads across archives can represent one recording. Those 438 entries carry 324 distinct 1024-byte prefixes. Normalized public handles provide an estimate of source diversity. One player can use multiple handles, and matching native names alone cannot establish identity.

The common external magic and version bytes are the same across all three patches. We decrypted the registered file prefix and decompressed the internal version fields. The minor suffix, major 0x0100 and embedded string such as `0100b` had to agree. Both nonempty stage offsets had to fit within the declared original body size. The native storage format includes the game binary's reported size and checksum. Four 1.00b entries report another checksum. [Preregistered category source records](catalogue-prefixes.json.gz) · [Counts and scope](summary.json) · [Reconstruction of the original writer](https://github.com/N0zoM1z0/th07/blob/2694946d17ea0fd1c219ffc255292ef9b79c4ade/src/ReplayManager.cpp)

The independent Sattori decoder read six complete candidate 1.00b files and two older-version controls. The report contains every public source, original SHA, decoded rank/shot, stage sequence, signed movement and firing evidence. One candidate emitted only three fire inputs per stage under the native name NoShot. The ordinary-play acquisition protocol therefore tracks five ordinary shortlisted recordings and one additional verified source, SEO. An observed original key/clock/camera match is required before any file produces training data. [Six initial candidates and two version controls](shortlist.json)

The optional Vpatch, translation and graphics components in the personal install remain untouched. A fresh isolated case uses read-only copies of the original executable and game data, a fresh private Wine prefix and score file, and one registered public input recording. No original binaries, save files or third-party replay files are distributed with this census.
