# FAQ

**Which method should I use?** `hybrid` unless you know your budget: `kmedian` for 1-10 images per class, `herding`
for about 50 or more.

**How many random networks (`n_nets`)?** One. In our ablation (K from 1 to 32) accuracy varied by at most 1.5 points.
The authors' main experiments used 8.

**Does the selection need labels?** Only to split by class; the features come from untrained networks.

**Why is `kcenter` bad?** It minimises the maximum distance, so with a small budget it picks outliers; see
[methods](methods.md).

**Can I use my own embeddings?** Yes: `select_coreset(None, labels, ipc, features=embeddings)` (an `(N, d)` array or a
list of arrays that are averaged as feature blocks).

**Results differ slightly from the documented numbers.** Those ran on Apple MPS; random networks and PAM tie-breaking can
differ across hardware. See [reproducing](reproducing.md).

**Is the selection deterministic?** Given the same seed and device, yes.
