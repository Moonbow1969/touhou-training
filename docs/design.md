# System design

## Native environment

The original Windows game supplies the simulation. The agent reads the live player position, resources, and nearby projectiles from memory. It applies keyboard input, advances the native game clock by four ticks, and reads the next state while the process is paused. At 60 native ticks per second, decisions run at about 15 Hz. Windows scheduling can add timing jitter, so replay labeling checks observed frame numbers and the controller records actual frames.

The structured policy receives normalized context and up to 96 nearby projectiles. Recent player movement comes from adjacent four-tick displacement intervals. The action heads predict nine movement directions including idle, plus binary focus, shoot, and bomb inputs. The original upstream pixel environment remains available for screenshot experiments.

## Demonstrations and models

Native replay files provide input words. State–action examples require playback and alignment against the original game. Audits record action-only frames, verified state–action decisions and world-model transitions as separate counts. A replay may contribute several stages, and stages and players remain grouped across splits. TH15, TH18 and TH19 are reserved for cross-title evaluation. The current training data covers a subset of the project’s planned titles, shot types and difficulties.

- **Policy SFT** learns actions from verified replay pairs. A soft movement target preserves changes inside each four-frame window. Validation includes moving and direction-change slices because idle and persistent actions can dominate overall accuracy.
- **World model** predicts a projectile occupancy grid and player displacement after 4, 12 or 24 native frames, conditioned on recorded inputs. Evaluation compares predicted occupancy with constant-velocity extrapolation. A controlled live action-ranking trial remains pending.
- **Native RL** currently fine-tunes a dodge head with PPO. The base SFT policy supplies a proposal, and the dodge head scores candidate movement and focus actions using analytic clearance. A separately supervised aim head can revise movement toward a vulnerable boss. The controller can model enemy hitboxes and straight lasers and issue emergency bombs.

The learned-only control benchmark starts from a human-replay SFT checkpoint. First run the frozen checkpoint with its actions applied directly. Then fine-tune the full policy with native RL and repeat the same evaluation. In both arms the policy outputs movement, focus, shoot and bomb. The published hybrid controller is the comparison baseline. Replay demonstrations are part of initialization, and the benchmark measures whether learned control can take over all four action heads.

## Evaluation

Native trials record build, mode, difficulty, shot, first miss, stage transition, bombs, score and termination reason. A one-credit result requires starting at Stage 1, using zero continues, and observing the game’s ending. Practice results and controller errors have their own outcomes. A native stage frame resets at each transition. First-miss times censored by a limit remain individually identifiable.

Keep the initial checkpoint, score rules, observation schema, difficulty, starting stage, and control settings fixed for paired ablations. Run the same controller with the imitation dodge head and with its PPO descendant. Compare the learned SFT proposal with a fixed shoot/focus baseline while keeping all downstream components equal. For the world model, compare a controller that uses its action ranking against the same controller using analytic clearance alone. Report clear rates alongside misses, bomb use and damage to avoid rewarding agents that survive by staying out of firing range.
