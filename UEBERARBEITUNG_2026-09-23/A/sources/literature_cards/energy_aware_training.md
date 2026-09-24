# Energy-Aware Training

Core idea: Optimize task loss with resource/activation penalties while enforcing accuracy constraints.

Transfer to QIRE: implemented as `energy_aware_loss` (`energy_loss.py`), wired
in via `--lambda-resource`/`--mu-activation`, but currently a no-op for
training dynamics: `resource_cost` (from `estimated_ops_per_sample()`) and
`activation_cost` (from `aux["active_model_parts"]`) are plain Python floats
computed only from static architecture config (e.g. `self.top_k`) in every
model class, never tensors connected to the autograd graph. Adding a constant
to the loss does not change its gradient, so `lambda_resource`/`mu_activation`
currently cannot push training toward sparser routing at all -- confirmed by
checking every model's `forward()` (`models_qire.py`, `models_efficient.py`,
`models_baseline.py`): `active_model_parts` is always a fixed float, never
computed from a differentiable quantity like soft gate probabilities.

Making the loss term differentiable is necessary but not sufficient here,
though: `gating.py`'s `entropy` (computed from `probs`, a real tensor with
gradients) is available and *could* be passed through undetached -- but it
would still not change measured resource cost for QIRE/StandardMoE as
architected, because `torch.topk` always activates exactly `top_k` experts
regardless of how peaked or flat the gate distribution is, and
`router_ops`/`expert_ops` are computed purely from the fixed hyperparameters
`num_experts`/`top_k`/`expert_hidden_dim` -- never from anything the gate
learns. So there is currently no trainable quantity in this architecture that
determines actual compute cost; `top_k` is a hyperparameter chosen and swept
via ablation before training, not something gradient descent can act on.
`EarlyExitMLP` (`models_baseline.py`) is the one model in this codebase where
compute usage genuinely is sample-dependent and trainable (it decides per
sample whether to use the early or full path based on a confidence
threshold) -- that is where an energy-aware loss term could have a real,
measurable effect. Making QIRE/StandardMoE's sparsity itself trainable (e.g. a
soft/thresholded top-k where low-probability experts are truly dropped rather
than always exactly `k` running) would be a substantial, genuinely novel
extension of the architecture, not a quick fix -- worth flagging as a
stretch goal if there is time, but should not be presented as already solved.

Candidate source (verify and read before citing): "Minimizing Energy
Consumption of Deep Learning Models by Energy-Aware Training" (EAT), arXiv:2307.00368
(2023) -- describes a differentiable L0-norm-style sparsity penalty on the
training loss; comparing its actual formulation against this codebase's
`energy_aware_loss` would show concretely what's missing here.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Do not cite until read.
