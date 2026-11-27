# Human direction prediction after original-game PPO

We evaluated the shared human SFT checkpoint and five PPO descendants on identical images from 12 previously used development stages. Each descendant was traced back to the human checkpoint by checking every intermediate deployed actor and trained parameter hash. The four-title set contains 33,265 decisions. These are predictions along human trajectories. Native rollouts separately measure whether the agent can survive the states reached by its own actions.

| Title | Decisions | All directions correct, SFT → six-title PPO | Direction changes correct, SFT → six-title PPO |
|---|---:|---:|---:|
| TH10 | 6,600 | 62.4% → 62.3% | 24.2% → 24.4% |
| TH12 | 9,806 | 53.8% → 53.5% | 22.3% → 22.0% |
| TH13 | 11,835 | 65.8% → 65.7% | 25.8% → 25.4% |
| TH17 | 5,024 | 45.2% → 44.9% | 22.4% → 21.9% |

Metrics pool native timing windows within each title. Existing held players were used repeatedly for development. Four additional TH17 players were reserved for a separate check. [All generations by title](previous-development/by-title.json) · [Per-stage scores and shifted-screen control](previous-development/held.json) · [Checkpoint registration](protocol.json)
