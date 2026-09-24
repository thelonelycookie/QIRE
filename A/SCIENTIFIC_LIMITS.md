# Scientific limits

QIRE AI Console v1.0 can run more experiments remotely, but remote execution does not prove efficiency.

Do not claim general QIRE efficiency unless:

- the result is reproduced over multiple seeds
- it holds on multiple datasets
- it is compared against fair baselines such as LogisticRegression, MLP, PrunedMLP and StandardMoE
- runtime, RAM and operation estimates are considered together
- smoke/exploration runs are separated from verification/final evaluation
- the claim is not contradicted by stronger classical baselines

Oracle Cloud runs are CPU-only in the intended free setup. Energy values remain estimates unless measured externally. QIRE router overhead must be reported separately from expert operations.
