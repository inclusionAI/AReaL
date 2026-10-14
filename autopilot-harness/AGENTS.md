# AutoPilot Harness contributor guide

- Read [README.md](README.md) for installation, usage, supported capabilities, and
  validation commands.
- Follow the repository-level [AGENTS.md](../AGENTS.md).
- Keep Harness code, configuration, examples, and tests in this package.
- Keep native training-runtime tests in the repository-level `tests/` directory.
- Document supported execution paths and distinguish CPU validation from GPU and
  distributed validation.
- Use synthetic inputs in committed examples and keep local run artifacts out of version
  control.
