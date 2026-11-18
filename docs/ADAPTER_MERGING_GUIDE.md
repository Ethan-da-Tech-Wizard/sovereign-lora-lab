# Architecture & Foundations: LoRA Adapter Merging & Task Arithmetic

## 1. The Challenge of Continual Edge Learning

When training an LLM on multiple specialized operational skills (e.g. system administration, tool parsing, code repair):
- Fine-tuning sequentially on separate datasets causes **catastrophic forgetting**.
- Joint multi-task fine-tuning from scratch requires massive consolidated datasets, complex hyperparameter tuning, and substantial GPU cluster compute.

With **Adapter Merging & Task Arithmetic**, you train independent, lightweight LoRA adapters ($r=16$, footprint $\sim 50\text{ MB}$ to $150\text{ MB}$ each) on focused domains, and then **merge, blend, or compress them into a unified adapter or base model** without retraining.

---

## 2. Comparison of Combination Strategies

| Technique | Description | Strengths | Trade-Offs |
|---|---|---|---|
| **Linear Task Arithmetic** | Weighted linear sum of task delta matrices | Instantaneous computation; great for compatible tasks | Can cause parameter collision if signs conflict |
| **Truncated SVD** | Low-rank matrix approximation via singular value decomposition | Keeps canonical rank $r=16$; zero rank expansion | Slightly higher compute than simple linear addition |
| **TIES Merging** | Trims low-magnitude weights and resolves sign conflict via majority voting | Eliminates destructive parameter interference | Requires density tuning parameter $\rho$ |
| **DARE Sparsification** | Drop-and-rescale random masking | Allows merging $5+$ adapters with minimal regression | Stochastic; requires fixed random seed |

---

## 3. Best Practices for Sovereign Edge Models

1. **Orthogonal Skill Separation:** Train adapters on distinct datasets (e.g. Adapter A = syntax fixes, Adapter B = CLI commands). Avoid training two adapters with conflicting system prompts on the exact same task.
2. **Normalized Learning Rates:** Ensure adapters are trained to loss plateau under similar optimizer schedules (e.g. cosine decay with warmup), so their delta gradient magnitudes are comparable.
3. **Task Weighting over Dataset Size:** Never weight adapters strictly proportional to sample size. A high-value skill with 100 high-quality verification pairs should often have higher weight than 5,000 generic chat pairs.
