# First PPO update with one original-game reset anchor

The frozen parent collected 13 qualified normal starts and 58 qualified continuations from the verified Stage 1 tick-2201 original anchor. The two sources supplied 11,540 and 10,278 decisions, respectively. The optimizer used every eligible anchor decision and sampled an equal number of ordinary decisions without replacement. Returns were computed before sampling the batch. Original first-spare-loss cutoffs retain a signed next camera and the parent critic's bootstrap value.

The RTX 4070 completed 430 accepted optimizer steps over two passes. Joint policy KL was 0.001461 and 0.001767 against a cap of 0.01. Critic RMSE went from 0.3565 to 0.3139 across the two fit epochs. The final TorchScript model returned finite logits for every action head on an original anchor observation.

The data emphasize a single restored region of Stage 1. Held restored starts measure performance from that original state. Promotion requires separately registered ordinary-start originals on the host.

[Training report](report.json) · [Source protocol and exact decisions](protocol.json) · [Original training source audit](../anchor-train)
