# TH08 v0.02 progress and spare-life reward

This is the reward definition and original evidence for the registered TH08 two-arm experiment.

Each actual policy action leads from one observed native state to the next. The base reward is playable native tick progress divided by the registered duration of that stage plus 1 for each verified new stage. Within a stage, equal ticks earn zero progress. Across a stage change, add the residual observed ticks in the old stage and the observed ticks in the new stage, each divided by its own duration. The boundary needs a verified last old-stage tick and first new-stage tick. A Continue, unverified skip, backwards within-stage clock or unreconciled spare ledger invalidates the credit. The length of a successfully completed stage can depend on player damage and dialogue. Freeze the normalization constants and state their physical source and coverage before an experiment. No TH08 duration constants are assumed by the module.

For the two arms, λ is 0 or 1. On a playable state with L spare lives, Φ = λ(L+1). Lost or earned spares change the potential. The active life remains represented with zero spares. At genuine Game Over or any other true objective terminal Φ = 0. A confirmed transition into Stage 2 on a continuing full-credit run retains the new state's full potential. Life or Bomb gauges sampled on loading/menu screens cannot substitute for the verified gameplay stocks. The upstream caller must supply independent stock-change events, stage evidence, and a no-Continue credit. The reported terminal tick is the last verified gameplay tick preceding the terminal screen.

With γ = 0.999 per decision, the shaped reward is r′ = r + γΦ(next) − Φ(current). For n decisions, summing discounted differences yields −Φ(start) + γⁿΦ(end). At a true terminal the boundary term vanishes. At a truncation with future gameplay, bootstrap using V′(end) = V(end) − Φ(end). This cancels the remaining boundary term. The returned bootstrap already includes that shift. A critic trained directly on shaped returns already estimates V′ and must not receive a second shift. The same shift at each value estimate leaves TD residuals and GAE unchanged when using corresponding critics. A common unshifted critic initialization can transiently change advantages through value error. A gain in finite-sample learning would require testing with held original credits. [Ng, Harada, and Russell on reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

Bombs, pickups, Power and score have reward weight zero. A separate optional 1% Bomb initialization is disabled. If later enabled, both arms would start from the identical initialized actor with its checkpoint hash logged. Binary on/off logits differing by log(0.01/0.99) give 1% when the feature-dependent contribution is zero. Adding that number to an already learned head generally gives a different probability. Record the actual initial probabilities and preserve direct sampling from all policy heads.

## Available source evidence

Four original Normal Ghost TRAIN credits in `th08-normal-current-online/native08/raw` were independently reopened. Seeds 80960101–104 contain 370, 600, 623, and 771 policy decisions. All began with two spares and three Bombs. All ended at the original Continue prompt with zero spares. The last playable Bomb counts were 3, 1, 3, 3. The screenshots, structured HUD and prompt observations, physical action masks, host monotonic capture times and host application times are available. Per-source digests and field names are in [observability.json](observability.json).

The original `th08tr.exe` SHA-256 is `3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1`. The independent replay observer in `tools/rollout-worker-probe/th08-trial002/camera.c` reads stage, manager tick, input words, focus, and native XY during original replay mode 1. The replay motion pilot verified 512 original Stage 1 camera pairs four native ticks apart. The four Normal live TRAIN archives expose no per-action gameplay stage/tick sequence. A post-credit ranking can indicate a reached stage, but cannot timestamp a specific policy transition. Their original host time, decision index and replay-only tick must not be converted into live progress labels. Before deployment, an independent live-mode fixture must verify stage-local monotonicity, transition bracketing, final playable ticks and original stock/terminal flags. No retroactive progress returns were assigned to these four credits.

## Checks

From the repository root, run `python -m unittest -v scripts.test_th08_progress_reward` and `python -m py_compile scripts/th08_progress_reward.py scripts/test_th08_progress_reward.py`. The focused checks cover telescoping and TD residual preservation, sparse positive stage reward, physical clock rollback, verified spare loss/gain and life exhaustion, a continuing Stage 2 truncation and an absorbing Stage 2 objective. The toy trajectories are arithmetic fixtures.

## Claude's next round

After the full warm-start readout results, Claude Opus 5.5 selected the Progress/Bomb pilot and the existing trained pooled SFT actor. A post-training check on 2,364 previously registered TRAIN decisions measured a mean Bomb probability of 0.0723%. The common initial policy shifts only the learned Bomb log-odds by +2.66049, equal to 1.0% across those recorded training views, with a zero critic. Both arms share this parent. The calibration inputs came from the previous historical cadence, so the new original-rollout log must report the actual sampled and consumed Bombs. The optional calibration described above has been enabled only for the new parent artifact.

The full original lifecycle engineering gate requires at least 99% consecutive four-tick decisions, 100% repeatable consumption of each applied mask, a complete stage and stock trace, and a real no-Continue endpoint. The first PPO update uses 16 fresh complete credits from the common behavior policy. Subsequent rounds require on-policy credits specific to each actor. Both arms use the same progress durations and the same initial policy, with λ equal to 0 or 1. The pilot caps each registered training epoch at 0.01 joint KL and the full study at three rounds. The held screening target is a life-loss rate ratio at most 0.75 and at most 60% of deaths while holding Bombs. A life-loss rate ratio at or above 1 is a stop. The experiment also records Bomb attempts and actual consumption per active life.

[Phase duration provenance](stage-duration.md) · [Bomb prior calibration](bomb-prior-diagnostic.json) · [Common behavior checkpoint](common-parent/policy.ts)

## First paired update completed

Sixteen full Normal Ghost credits supplied 14,447 consumed decisions. The sealed original observed 14,409 of 14,431 complete four-tick intervals. Every admitted decision had its intended native mask consumed. The common training actor lost 48 lives, with Bomb stock remaining at 31 of those events. The original recorded 75 verified same-life Bomb stock reductions consuming 78 units. Both learners verify the rollout against the deployed CPU actor before GPU optimization.

| Training arm | Accepted updates | Final joint KL to shared behavior |
|---|---:|---:|
| Original progress | 302 | 0.00213718 |
| Original progress plus spare-life potential | 302 | 0.00221052 |

The arms learned from the same original decisions and share all sampling and optimizer parameters. All four attempted epochs passed the 0.01 KL cap. Critic losses use the respective training returns, which are on different scales. The independent original-game evaluation measures the actual impact.

[Original training collection](original-collection/) · [First update protocol](first-paired-update/protocol.json) · [Complete learner log](first-paired-update/learner.log) · [Progress checkpoint](first-paired-update/progress/policy.ts) · [Progress-plus-spares checkpoint](first-paired-update/progress-plus-spares/policy.ts)

Claude Opus 5.5 reviewed both accepted updates on November 27 and released the six previously registered held pairs. The primary denominator is observed native ticks and the count is verified life losses. The reserve-potential candidate qualifies for continuation when its life-loss rate ratio to progress alone is at most 0.75 and no more than 60% of its deaths occur while retaining Bombs. A rate ratio at or above 1.0 stops the current-scale RL experiment. Stage 2 and terminal resources are secondary outcomes. The shared training cohort is descriptive context for the calibrated parent. An independent held parent would be required to measure improvement over that parent. [Held registration and metric definitions](held-original/)

## Independent original evaluation completed

The original game confirmed all twelve Stage 1 Game Over endpoints, with zero continues. The progression arm had 18 life losses over 18,514 native ticks, and the spare-potential arm had 18 over 19,393. These exposure totals give 0.972 and 0.928 deaths per 1,000 ticks, a rate ratio of **0.955**. The six-pair descriptive bootstrap range is **0.730–1.320**. Both arms had **15/18** losses with Bombs in stock and consumed 23 Bomb units. Mean final Bomb stock was 1.00 and 1.17, both with zero spare lives and 0/6 reaching Stage 2.

Only three complete pairs passed strict control verification in both arms. Their life-loss rate ratio was 0.862, and each arm had 8/9 losses with Bombs available. These data fail the registered promotion gate. The original endpoints are retained in the full denominator. Several interrupted native controls prevent a clean full-cohort reward comparison.

The longest logged native read was 1.630 seconds. The three longest read durations coincide with synchronous recording snapshots from the original game. An isolated offline implementation preserves final input bytes while moving archive persistence away from the clock reply. Live latency validation remains open. [Full native evaluation](held-original/) · [Asynchronous archive verification](observer-archive-prototype/)

Claude reviewed the held results and selected exactly two original engineering credits for the next native writer revision. Both require at least 99.8% exact four-tick intervals, observer round-trip P99.9 below 30 ms, no read above 100 ms, and complete original video, input and stock evidence. The reward arms are archived. A larger follow-up on λ=1 requires the writer gate and a new budget review.
