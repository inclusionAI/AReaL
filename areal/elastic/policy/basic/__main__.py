# SPDX-License-Identifier: Apache-2.0

"""CPU-only entry point: python areal/elastic/policy/basic scenario.json."""

if __package__:
    from .cli import main
else:
    import importlib.util
    import sys
    from pathlib import Path

    # Import the Elastic namespace without areal's eager training dependencies.
    directory = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_areal_elastic",
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from _areal_elastic.policy.basic.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
