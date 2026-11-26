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

## First state-only student to reach Stage 2

A single frozen network received current native state and observable times since encounter events. Its three output heads select joint movement/focus, Shoot and Bomb. Its teacher used a training replay; inference receives no replay trajectory. On its first original stock credit, seed 2601, it reached Stage 2 with one spare life and no Bombs and ended at Stage 2 frame 8,555 with zero Continues. It lost four lives and used 14 Bombs over the credit. This is a single result for a distilled checkpoint prior to online PPO. The locally recorded 5:11.70 MP4 has SHA-256 `77288685acc4605167388b1762c0d01b9b69c7508380d4c2d12d2c781fb76c28`. [Full credit](first-state-only-stage2-2601.json) · [Training log](relative-student-training.json) · [Reserved native state agreement](relative-student-2402-holdout.json)

The movement head learned a waypoint from stage and encounter state and a motor correction from the current player position. Its decoded movement did not directly use the supplied projectile rows. The new full-credit PPO recipe connects a learned projectile-risk representation through a trainable residual. The residual starts at zero. Replaying all 4,553 archived decisions reproduced all four original controls exactly. At training temperature 0.55 the expected joint move/focus agreement on the same states was 93.55%. [Initialization contract](full-policy-ppo-initialization-2601.json)

A separate demonstration-conditioned pilot, seed 2402, cleared Stage 1 with zero spare lives and zero Bombs and died at Stage 2 frame 508, before encountering its modified Stage 2 boss clock. The three lethal contacts in this run were physically consistent with a retained active projectile. [Outcome](exact-stage2-2402-story.json) · [Passive hazard audit](demo-exact-stage2-2402-sensor-audit.json) · [Projected opening clearance](stage2-opening-projected-clearance.png)

The first 256-decision final-spell suffix attempt completed collection and optimization, then failed checkpoint validation because JSON converted the prefix's integer stage keys into strings. No weights were recovered from that attempt. The retry normalizes metadata before entering the game; a full SB3 archive passed staged save and read-back. [Interrupted run](final-spell-first-checkpoint-failure.json)

## Full-policy online PPO, first frozen comparison

Two stochastic full-life credits from the state-only student generated 5,021 original-game decisions. Both reached Stage 2 before the first optimizer update, with 0/2 and 0/0 spare lives/Bombs. Both rollouts finished at the training target. The updated policy changed 37 of 4,553 greedy move/focus decisions on the separate archived seed-2601 student trajectory. Shoot and Bomb decisions matched throughout. [Native training log](state-only-native-ppo-5021-training.json) · [Policy drift](state-only-ppo-5021-drift.json)

Both frozen controllers then played independent fresh credits to Game Over with policy seed 3601. PPO reached Stage 2 with zero spares and three Bombs and died at Stage 2 frame 7,995. Its frozen parent entered with zero spares and two Bombs and died at frame 5,298. The game RNG is uncontrolled. The two credits establish an observed difference of 2,697 stage ticks. Additional credits are required to estimate a typical difference. Both terminal losses occurred with empty Bomb stock. [PPO credit](state-only-ppo-5021-ending-3601.json) · [Frozen parent](state-only-frozen-ending-3601.json) · [Full-life comparison figure](figures/stock-stage2-rates.png)

The separate 256-decision late-spell PPO checkpoint passed the native training save and restore. A fresh stock prefix supplied the final-spell states. Two learner-controlled attempts ended in Game Over, a third learner segment reached the collection budget, and another prefix died before the learner took control. [Training report](final-spell-native-ppo-256.json)

## Action choice in the official trial

Six original TH14 0.01b instances assigned 18 move/focus options uniformly in near-projectile states. There were 171 confirmed injuries in 4,272 held 12-tick windows. A replay-trained model scored the actual action at average precision 0.480 versus 0.237 for the mean score of the nine moves at the same focus. [Native randomized validation](anticipatory-contact-trial-prospective-expanded.json)

On this archived cohort, uniform actions incurred 4.00% injury risk. An inverse-propensity estimate for sampling uniformly among the four lowest-risk moves at a uniformly chosen focus was 1.89%. The paired difference was −2.11 percentage points (95% lane-stratified episode bootstrap interval −2.70 to −1.58). This estimates one held action on the state distribution generated by uniform play. Repeated use changes subsequent game states and requires a separate closed-loop test. [Randomized-window policy estimate](trial-randomized-action-policy-ips.json)
