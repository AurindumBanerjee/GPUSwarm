# 05 Continuous relaxation + gradient polish

At the end of each unfinished stage: ports fixed at gbest, each cap
relaxed to continuous (log R, log L, log C), Adam on a smooth p-norm peak
(`--steps`, `--lr`, `--pnorm`), then snap to the `--snap-k` nearest real
parts with a greedy exact (table) coordinate pass. Accepted only if it
beats gbest; polish time and evaluations are counted.

Needs `--freq`. `polish_stats.json`: calls, improvements, mean gain.

Methods: numpy, fast, fast_polish.
