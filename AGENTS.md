# Permanent Project Rules

- C++17 and Python 3.11 are the supported language levels.
- CPU functionality must remain usable without a GPU.
- CUDA, HIP and MLIR integrations must be optional feature flags.
- Never install packages or download dependencies unless the user explicitly requests it.
- Never fabricate test output, benchmark output, hardware output or performance metrics.
- Never mark GPU work complete unless it was executed on real matching hardware.
- Never silently replace a requested implementation with a mock.
- Fail explicitly for unsupported operators, shapes and dtypes.
- Use deterministic seed 42 for reproducible fixtures.
- Use RAII and prohibit owning raw pointers.
- Validate integer multiplication and allocation sizes before allocating memory.
- All public behaviour must be covered by automated tests.
- Every milestone must create `docs/phase_reports/phase_NN.md`.
- Complete only the requested milestone and stop.
- Existing public contracts may not be changed without migration tests.
- No performance claim may be written until a reproducible benchmark generated it.
- Critical paths may not contain TODO, pass, NotImplementedError or fake return values.
- At the end of every milestone run all relevant tests and document the exact commands and outcomes.
