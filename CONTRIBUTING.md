# Contributing

Bug reports, fixes, new selectors and benchmarks are welcome.

```bash
pip install -e ".[dev]"
pytest -m "not slow"        # fast CPU tests (about 15 s)
pytest                      # all tests
ruff check src tests
```

Guidelines
* Every selector needs a test against an *independent* reference (brute force, exact linear program, or a known
  property), not only shape checks. The greedy facility-location test compares with an explicit marginal-gain loop.
* Keep the public API stable: `select_coreset`, `CoresetResult`, the solvers in `coreset_ot.solvers`.
* New results claimed in the documentation must come with the command that produces them.
