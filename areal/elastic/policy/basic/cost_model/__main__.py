# SPDX-License-Identifier: Apache-2.0

"""Also supports `python areal/elastic/policy/basic/cost_model` without training dependencies."""

if __package__:
    from .cli import main
else:
    # Loading this standalone package avoids areal.__init__'s eager training imports.
    # It does not inject a fake areal package or change the installed training API.
    import importlib.util
    import sys
    from pathlib import Path

    directory = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_areal_elastic",
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from _areal_elastic.policy.basic.cost_model.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
