# 03 Incremental Woodbury updates (refactorisation analogue)

If a particle keeps its ports and only changes some capacitor models, the
stored inverse of M = Z0[P,P] + diag(1/d) is updated with rank-1
Sherman-Morrison steps (O(k^2) per frequency) instead of re-factorised.
Full re-inverse when ports change, more than `--max-rank` models change,
or after `--refresh` consecutive updates. Drift vs fresh inverses is
measured every `--drift-check` calls.

Methods: numpy, woodbury_batch, incr.
