# TH14 Normal full-credit policy development

## Randomized Bomb decisions

Four Reimu A story credits provided 31 verified 16-frame windows. A random assignment at the beginning of each enrolled window chose press or release with equal probability. The remaining three four-frame decisions released Bomb. Enrollment required positive native inventory. The frozen PPO-397 actor supplied movement, focus and shooting.

| Native player state | Release | Press |
| --- | --- | --- |
| Ordinary state 1 | 0 life losses in 5 windows | 0 life losses in 8 windows, 7 Bombs consumed |
| Hit state 4 | 8 life losses in 8 windows | 0 life losses in 10 windows, 10 Bombs consumed |

The table and a separate linear two-action value model were fitted using native player state, previous Bomb key and inventory. This observation contract adds information to the original PPO actor. The table requires support from both randomized actions and uses the anchor policy for unsupported entries. The linear model predicts continuously from its fitted coefficients. Stock zero and previous-key-held contexts had no randomized support. [Interventions](bomb-randomized-collection.json) · [Fitted table](bomb-effect-table.json) · [Linear fit](bomb-value-fit.json)

| Policy sample seed | PPO-397 Game Over | PPO-397 + fitted table Game Over | Actual Bombs consumed, control / fitted |
| --- | ---: | ---: | ---: |
| 1701 | 6317 | 10973 | 1 / 9 |
| 1702 | 5467 | 10114 | 0 / 9 |

All four full credits ended in Stage 1. The fitted head passed the boss spell with native ID 4 in both trials. Stage 2 arrivals were 0/2 in each arm. Fixed policy sampling seeds do not control native game randomness. [Native paired evaluation](bomb-table-eval.json)

## Where the original late-stage curriculum failed

The fixed learned crossover was scheduled to hand control to an online PPO learner at frame 7600. It died at 7482, 7406, 5270 and 6170 in the four attempted prefixes. No learner transitions or updates occurred. A subsequent run selected frame 5000. [Rejected prefixes](rejected-7600-prefix.json) · [Reference action trace](reference-steps.csv)

## Testing the movement anchor

The same fitted linear Bomb value function was applied on every stage and every decision. Its base action came from PPO-397 or independently trained PPO-211. All four complete credits ended in Stage 1. The Game Over frames on policy seeds 1901 and 1902 were 10,168 and 10,069 for PPO-397, and 8,284 and 8,268 for PPO-211. Both movement actors exhausted all three starting lives on both seeds. These trials were scored at ordinary stock speed. A failed menu-image verification between credits required reacquisition of a verified new Stage 1 credit before the remaining trials. [Native results](bomb-value-story-transfer.json)

The retained native observation/action archives contain full boss HP and vulnerability. On 1,750 right-wall observations of the first PPO-397 credit, inward movement had mean probability 0.0152 under PPO-397, 0.3743 under PPO-211 and 0.0909 under their 50:50 parameter interpolation. These are per-observation probabilities. No native clearance was measured for the interpolated actor. [Wall probe](native-wall-probes.json)

The frozen human-data TH14 training split had 85,416 rows and 591 Bomb-positive four-tick chunks. All 85,416 training rows excluded the native hit state due to the movement-controllability filter. Future captures can retain aligned hit and recovery windows separately. A raw cadence-rejected archive was used only to verify extraction. [Input alignment audit](human-hit-window-audit.json)

## Online PPO and demonstration-conditioned credits

After 512 native suffix decisions, the same learner resumed for 4,096 more at a frame-5000 handoff. Nine suffix episodes ended at Game Over in the continuation. Stage 2 entries were 0/9. The checkpoint contains the optimizer state. Policy-seed-blocked comparisons against the initializer and the earlier checkpoint are tracked separately from these training episodes. [Training record](boss-ppo-4608.json)

A neural tracker followed a six-stage training demonstration from a verified Stage 1 Normal Reimu A opening, with a fitted Bomb model and zero Continues. The four-tick actor reached Game Over at frame 9,761. During the final spell, its reference had already ended. It requested fire on 0/207 decisions, leaving the boss at 1,891 of the 1,900 HP observed at spell onset. The two-tick variant reached Game Over at 10,715 and consumed 10 Bombs. Its active stage and global timers both advanced by exactly two on 3,998/4,002 decisions. Each cadence has one credit and a different policy seed. [Four-tick credit](demo-route-four-tick-2001.json) · [Phase and damage trace](demo-route-first-native-2001-audit.json) · [Two-tick credit](demo-route-two-tick-2101.json)

## First Stage 2 entry with the phase-aligned demonstration

The first revised credit (seed 2201) cleared Stage 1 with one spare life and zero Bombs, having consumed seven Bombs and lost one life. On the final Stage 1 spell it requested fire on all 273 vulnerable decisions. Sampled boss HP went from 1,900 to zero. The preceding four-tick credit requested fire on 0/207. The revision changes both phase alignment and Shoot selection. A single comparison does not identify their separate effects. The frozen old SFT supplies the independent Shoot head. The movement tracker still uses an explicit training demonstration at inference and the Bomb head uses native player state.

The uninterrupted credit ended in Stage 2 at native frame 9,452 with zero Continues, five total lives lost and 17 actual Bombs consumed. The Stage 2 score includes resources earned during play. [Story report](first-phase-aligned-story-2201.json) · [Phase telemetry](first-phase-aligned-telemetry-2201.json) · [Measured final-spell trace](stage1-final-spell-comparison.png)

The same frozen phase-aligned actor entered Stage 2 on a second full credit (seed 2202). It again carried one spare life and zero Bombs, then reached Game Over at Stage 2 frame 8,725. The first two eligible credits give 2/2 Stage 2 entries and 0/2 Stage 3 entries. The Wilson 95% interval for Stage 2 is 34.2–100%, assuming independent credits. Actual game RNG is uncontrolled. Both ended with an empty Bomb stock. [Replication](replicated-phase-aligned-story-2202.json) · [Cohort](phase-aligned-two-credit-cohort.json) · [Resource ledger](phase-aligned-resource-ledger.json)

In the first matched frame-5000 suffix block, the single-network initializer, 512-step checkpoint and 4,608-step checkpoint reached Game Over at native frames 6,983, 4,970 and 7,742. The 512-step arm died during the shared prefix and never took control. All three full credits count in the denominator. Different native game trajectories prevent attribution of the terminal-frame differences to the suffixes. [Matched block](boss-suffix-seed2401-pairing.json) · [Raw credits](boss-suffix-seed2401-stock.json)

A 12-tick neural contact predictor trained on 108,329 human-replay decisions, using 18 hypothetical held commands per decision. It was selected on 2,205 rows from distinct validation replays. On seven separately recorded native credits, there were 199 positive overlapping windows among 8,083 eligible held-action windows. Action-conditioned average precision was 0.504, versus 0.386 when averaging directions at the observed focus and 0.368 for the action-agnostic head. The endpoint is a native injury within 12 ticks. The training targets use projected circular contacts on exogenous bullet fields. No original-game avoidance-rate improvement is inferred from this classifier measurement. At six of nine observed lethal events, the nominal reference position itself overlapped a projected active projectile. Two further events were unexplained by retained circular bullets. [Predictor measurements](anticipatory-contact-12tick.json) · [Per-event trajectory counterfactuals](phase-route-fatal-hit-margins.json)
