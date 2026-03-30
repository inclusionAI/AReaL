"""Functions for registering environments within skyrl_gym using public functions ``make``, ``register`` and ``spec``."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
from dataclasses import dataclass, field
from agent_system.environments.env_package.search.third_party.skyrl_gym import Env, error

from typing import Protocol, Dict, Any

__all__ = [
    "registry",
    "EnvSpec",
    # Functions
    "make",
    "spec",
    "register",
    "pprint_registry",
]


class EnvCreator(Protocol):
    """Function type expected for an environment."""

    def __call__(self, **kwargs: Any) -> Env: ...


@dataclass
class EnvSpec:
    """A specification for creating environments with `skyrl_gym.make`.

    * **id**: The string used to create the environment with `skyrl_gym.make`

    * **entry_point**: A string for the environment location, ``(import path):(environment name)`` or a function that creates the environment.
        NOTE[shu]: example is like a database path? or local RAG environment?

    * **kwargs**: Additional keyword arguments passed to the environment during initialisation

    """

    id: str
    entry_point: EnvCreator | str | None = field(default=None)

    # Environment arguments
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # post-init attributes
    name: str = field(init=False)

    def __post_init__(self):
        """Calls after the spec is created to extract name from the environment id."""
        self.name = self.id

    def make(self, **kwargs: Any) -> Env:
        """Calls ``make`` using the environment spec and any keyword arguments."""
        return make(self, **kwargs)

    def to_json(self) -> str:
        """Converts the environment spec into a json compatible string.

        Returns:
            A jsonifyied string for the environment spec
        """
        env_spec_dict = dataclasses.asdict(self)
        env_spec_dict.pop("name")

        # To check that the environment spec can be transformed to a json compatible type
        self._check_can_jsonify(env_spec_dict)

        return json.dumps(env_spec_dict)

    @staticmethod
    def _check_can_jsonify(env_spec: Dict[str, Any]):
        """Warns the user about serialisation failing if the spec contains a callable.

        Args:
            env_spec: An environment.

        Returns: The specification with lambda functions converted to strings.

        """
        spec_name = env_spec["name"] if "name" in env_spec else env_spec["id"]

        for key, value in env_spec.items():
            if callable(value):
                raise ValueError(
                    f"Callable found in {spec_name} for {key} attribute with value={value}. Currently, skyrl_gym does not support serialising callables."
                )

    @staticmethod
    def from_json(json_env_spec: str) -> EnvSpec:
        """Converts a JSON string into a specification stack.

        Args:
            json_env_spec: A JSON string representing the env specification.

        Returns:
            An environment spec
        """
        parsed_env_spec = json.loads(json_env_spec)

        try:
            env_spec = EnvSpec(**parsed_env_spec)
        except Exception as e:
            raise ValueError(f"An issue occurred when trying to make {parsed_env_spec} an EnvSpec") from e

        return env_spec

    def pprint(
        self,
        disable_print: bool = False,
        include_entry_points: bool = False,
    ) -> str | None:
        """Pretty prints the environment spec.

        Args:
            disable_print: If to disable print and return the output
            include_entry_points: If to include the entry_points in the output
            print_all: If to print all information, including variables with default values

        Returns:
            If ``disable_print is True`` a string otherwise ``None``
        """
        output = f"id={self.id}"
        if include_entry_points:
            output += f"\nentry_point={self.entry_point}"

        if disable_print:
            return output
        else:
            print(output)


# Global registry of environments. Meant to be accessed through `register` and `make`
registry: Dict[str, EnvSpec] = {}


def _find_spec(env_id: str) -> EnvSpec:
    # For string id's, load the environment spec from the registry then make the environment spec
    assert isinstance(env_id, str)

    # load the env spec from the registry
    env_name = env_id
    env_spec = registry.get(env_name)

    if env_spec is None:
        raise error.Error(
            f"No registered env with id: {env_name}. Did you register it, or import the package that registers it? Use `skyrl_gym.pprint_registry()` to see all of the registered environments."
        )

    return env_spec


def load_env_creator(name: str) -> EnvCreator:
    """Loads an environment with name of style ``"(import path):(environment name)"`` and returns the environment creation function, normally the environment class type.

    Args:
        name: The environment name

    Returns:
        The environment constructor for the given environment name.
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def _check_spec_register(testing_spec: EnvSpec):
    """NOTE[shu]: Checks that no environment with the same name already exists in the registry."""
    for env_spec in registry.values():
        if env_spec.name == testing_spec.name:
            raise error.RegistrationError(
                f"An environment with name `{testing_spec.name}` is already registered (id=`{env_spec.id}`). "
                "Environment names must be unique."
            )


def register(
    id: str,
    entry_point: EnvCreator | str | None = None,
    kwargs: Dict[str, Any] | None = None,
):
    """
    Registers an environment in skyrl_gym with an ``id`` to use with `skyrl_gym.make` with the ``entry_point``
    being a string or callable for creating the environment.

    The ``id`` parameter corresponds to the name of the environment.

    It takes arbitrary keyword arguments, which are passed to the :class:`EnvSpec` ``kwargs`` parameter.

    Args:
        id: The environment id
        entry_point: The entry point for creating the environment
        kwargs: arbitrary keyword arguments which are passed to the environment constructor on initialisation.
    """
    assert entry_point is not None, "`entry_point` must be provided"
    global registry

    if kwargs is None:
        kwargs = dict()

    new_spec = EnvSpec(
        id=id,
        entry_point=entry_point,
        kwargs=kwargs,
    )
    _check_spec_register(new_spec)
    registry[new_spec.id] = new_spec


def make(
    id: str | EnvSpec,
    **kwargs: Any,
) -> Env:
    """Creates an environment previously registered with `skyrl_gym.register` or a :class:`EnvSpec`.

    To find all available environments use ``skyrl_gym.envs.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id or a :class:`EnvSpec`.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist in the `registry`
    """
    if isinstance(id, EnvSpec):
        env_spec = id
    else:
        # For string id's, load the environment spec from the registry then make the environment spec
        assert isinstance(id, str)

        # The environment name can include an unloaded module in "module:env_name" style
        env_spec = _find_spec(id)

    assert isinstance(env_spec, EnvSpec)

    # Update the env spec kwargs with the `make` kwargs
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    env_spec_kwargs.update(kwargs)

    # Load the environment creator
    if env_spec.entry_point is None:
        raise error.Error(f"{env_spec.id} registered but entry_point is not specified")
    elif callable(env_spec.entry_point):
        env_creator = env_spec.entry_point
    else:
        # Assume it's a string
        env_creator = load_env_creator(env_spec.entry_point)

    try:
        env = env_creator(**env_spec_kwargs)
    except TypeError as e:
        raise type(e)(f"{e} was raised from the environment creator for {env_spec.id} with kwargs ({env_spec_kwargs})")

    if not isinstance(env, Env):
        if str(env.__class__.__base__) == "<class 'skyrl_gym.core.Env'>":
            raise TypeError(
                "Gym is incompatible with skyrl_gym, please update the environment class to `skyrl_gym.Env`. "
            )
        else:
            raise TypeError(f"The environment must inherit from the skyrl_gym.Env class, actual class: {type(env)}. ")

    # Set the minimal env spec for the environment.
    env.spec = EnvSpec(
        id=env_spec.id,
        entry_point=env_spec.entry_point,
        kwargs=env_spec_kwargs,
    )

    return env


def spec(env_id: str) -> EnvSpec:
    """Retrieve the :class:`EnvSpec` for the environment id from the `registry`.

    Args:
        env_id: The environment id with the expected format of ``[(namespace)/]id[-v(version)]``

    Returns:
        The environment spec if it exists

    Raises:
        Error: If the environment id doesn't exist
    """
    env_spec = registry.get(env_id)
    if env_spec is None:
        raise error.Error(f"No registered env with id: {env_id}")
    else:
        assert isinstance(
            env_spec, EnvSpec
        ), f"Expected the registry for {env_id} to be an `EnvSpec`, actual type is {type(env_spec)}"
        return env_spec


def pprint_registry(
    print_registry: Dict[str, EnvSpec] = registry,
    *,
    num_cols: int = 3,
    disable_print: bool = False,
) -> str | None:
    """Pretty prints all environments in the registry without grouping by namespace.

    Args:
        print_registry: Environment registry to be printed. By default, uses the global `registry`.
        num_cols: Number of columns to arrange environments in.
        disable_print: Whether to return a string instead of printing it.
    """
    # Get all environment ids
    env_ids = sorted(print_registry.keys())

    if not env_ids:
        output = "No environments registered."
        if disable_print:
            return output
        else:
            print(output)
            return

    # Find the max width for nice column alignment
    max_justify = max(len(env_id) for env_id in env_ids)

    # Build the output
    output_lines = []
    current_line = ""

    for count, env_id in enumerate(env_ids, 1):
        current_line += env_id.ljust(max_justify + 2)

        if count % num_cols == 0 or count == len(env_ids):
            output_lines.append(current_line.rstrip())
            current_line = ""

    final_output = "\n".join(output_lines)

    if disable_print:
        return final_output
    else:
        print(final_output)
