# SkyRL-Gym

A library of RL environments for LLMs implemented with the Gymnasium API.

## Key Features

- Simple `Environment` interface following the Gynasium API. 
- Library of ready-built environments for math, code, search, and text-to-SQL.
- A reusable `tool` interface. Developers can implement a tool once, and use it across any environment.
- Supports multi-tool environments

## Installation

You can install the latest release from PyPI:

```bash
pip install skyrl-gym
```

or install from source:

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-gym
pip install -e .
```

## Documentation

To build your first environment, see our [Walkthrough Docs](https://skyrl.readthedocs.io/en/latest/tutorials/new_env.html).

All docs are available at [https://skyrl.readthedocs.io/en/latest/](https://skyrl.readthedocs.io/en/latest/).
