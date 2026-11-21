# PPO defenders from the B-ACE 2.0 paper

Independent PPO (shared weights) trained with `python -m bace.ppo` in the *defend* scenario
(blue DCA against red strikers flying Duck and the three reference pilots), 4M agent
decisions each, checkpoint chosen on validation seeds by `bace.ppo.select_checkpoint`.

| File | Team size | Seed |
|------|-----------|------|
| `1v1_s0.pt`, `1v1_s1.pt`, `1v1_s2.pt` | 1v1 | 0, 1, 2 |
| `2v2_s0.pt`, `2v2_s1.pt`, `2v2_s2.pt` | 2v2 | 0, 1, 2 |

```python
import torch
from bace.ppo import Policy, evaluate_vec

ck = torch.load("models/ppo_defend/1v1_s0.pt", weights_only=False)
print(evaluate_vec(ck, "ref_balanced", list(range(40_000, 40_200))))
act = Policy(ck)          # obs -> MultiDiscrete[2, 5, 5] action (sampled, as trained)
```

Each file holds the actor, critic, observation normaliser and training config.
`python -m bace.studies rl_eval --policies models/ppo_defend/*.pt` reproduces the paper's
held-out evaluation.
